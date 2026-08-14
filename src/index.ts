/**
 * Exploration-fold plugin: registers `ctx.rewind` (a service that folds
 * everything since the most recent `checkpoint/mark` into an auto-generated
 * report) and the model-facing `rewind` tool over it. The fold shadows the
 * exploration's surface range with the report — the same `surfaceOp:
 * { op: 'replace' }` mechanism the compaction seam uses — so subsequent
 * requests no longer carry the exploration's intermediate steps, while the
 * durable log keeps the full exploration for audit. Consumes `ctx.llm` for
 * the report and the compaction seam's exported pair-balance helpers; never
 * modifies upstream packages.
 *
 * @module @dpskh/tool-rewind
 */

import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { createUserMessage, errorChain } from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import * as toolPlugin from './tool.ts'

// Re-export pulls the declaration-merged event map into every consumer
// program (the dsh-compact pattern): importing this package is what makes
// the 'checkpoint/*' events visible to `Session.append` and `SessionEvent`.
export type { CheckpointRewindEvent } from './types.ts'

import { REWIND_REPORT_SOURCE } from './brand.ts'
import { installRewindGuard } from './guard.ts'
import {
  RewindError,
  assertNoActiveFold,
  assertRegionUnchanged,
  findLatestMark,
  openTurnOf,
  selectFoldRegion,
} from './region.ts'
import {
  buildSummarizationInput,
  regionTextLength,
  summarizeRegion,
  type RewindSummarizeConfig,
} from './summarize.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    rewind: RewindService
  }
}

/** One settled fold, as reported to the caller. */
export interface RewindResult {
  /** Seq of the `checkpoint/mark` the fold started after. */
  checkpointSeq: number
  /** Number of surface nodes folded into the report. */
  foldedNodes: number
  /** Inclusive first folded surface-node seq. */
  start: number
  /** Inclusive last folded surface-node seq. */
  end: number
}

/**
 * The exploration-fold service, registered as `ctx.rewind`. One instance per
 * context; the verb runs the full fold transaction: find the latest mark,
 * select the balanced region, summarize it over `ctx.llm`, then commit the
 * replacement in one synchronous block. Failures leave the fold bracket open
 * with an error record (failed attempts remain visible in the log).
 */
export class RewindService extends Service {
  static inject = ['llm']

  private readonly maxTokens: number
  private readonly maxSummarizationRetries: number
  private readonly summarizationProvider: string | undefined
  private readonly summarizationModel: string | undefined
  private readonly reportLanguage: 'en' | 'zh'

  constructor(ctx: Context, config: Partial<RewindSummarizeConfig> = {}) {
    super(ctx, 'rewind')
    this.maxTokens = config.maxTokens ?? 1024
    this.maxSummarizationRetries = config.maxSummarizationRetries ?? 2
    this.summarizationProvider = config.summarizationProvider
    this.summarizationModel = config.summarizationModel
    this.reportLanguage = config.reportLanguage ?? 'en'
  }

  /**
   * Fold everything since the most recent `checkpoint/mark` into an
   * auto-generated report. The report replaces the folded surface range in
   * the model-visible history; the durable log keeps the exploration.
   * @param agent - the agent whose session to fold.
   * @param signal - cancellation signal forwarded to the summarization.
   * @returns the settled fold result.
   * @throws {@link RewindError} for expected no-checkpoint, empty-region,
   * unbalanced, busy-lock, changed-span, and shrink failures, and the exact
   * abort reason when cancelled.
   */
  async rewind(agent: Agent, signal: AbortSignal): Promise<RewindResult> {
    const session = agent.session
    const events = session.events
    const mark = findLatestMark(events)
    if (mark === undefined) {
      throw new RewindError('NO_CHECKPOINT', 'rewind: no checkpoint/mark found in the session; call `checkpoint` before exploring')
    }
    const region = selectFoldRegion(session, mark.seq)
    assertNoActiveFold(events)
    const turn = openTurnOf(events)
    session.append('checkpoint/fold-start', { turn })
    let closed = false
    try {
      const input = buildSummarizationInput(session, region)
      const summary = await summarizeRegion(this.ctx, input, agent, {
        maxTokens: this.maxTokens,
        maxSummarizationRetries: this.maxSummarizationRetries,
        ...this.summarizationProvider === undefined || this.summarizationModel === undefined
          ? {}
          : { summarizationProvider: this.summarizationProvider, summarizationModel: this.summarizationModel },
        reportLanguage: this.reportLanguage,
      }, signal)
      signal.throwIfAborted()
      assertRegionUnchanged(session, mark.seq, region)
      const foldedText = regionTextLength(input.messages)
      if (foldedText > 0 && summary.report.length >= foldedText) {
        throw new RewindError(
          'SHRINK',
          `rewind summary (${summary.report.length} chars) is not smaller than the folded region (${foldedText} chars)`,
        )
      }
      // Synchronous commit: no await between the record and the bracket close.
      const record = session.append('checkpoint/rewind', {
        turn,
        report: summary.report,
        checkpointSeq: mark.seq,
        shadowedRange: { start: region.start, end: region.end },
        shadowedSeqs: [...region.shadowedSeqs],
        provider: summary.provider,
        model: summary.model,
        ...summary.usage === undefined ? {} : { usage: summary.usage },
      })
      session.append('user/message', createUserMessage({
        content: [{ type: 'text', text: summary.report }],
        source: REWIND_REPORT_SOURCE,
      }), {
        surfaceOp: { op: 'replace', start: region.start, end: region.end },
        sourceEventSeqs: [record.seq, ...region.shadowedSeqs],
      })
      session.append('checkpoint/fold-end', { turn })
      closed = true
      return {
        checkpointSeq: mark.seq,
        foldedNodes: region.shadowedSeqs.length,
        start: region.start,
        end: region.end,
      }
    } catch (error) {
      /* v8 ignore next 1 -- closed stays false on every throwing path: nothing after the commit block can throw */
      if (!closed) {
        try {
          session.append('checkpoint/fold-end', { turn, error: errorChain(error) })
        } catch {
          // The bracket close itself failed; the unmatched fold-start stays
          // detectable as a fold lock.
        }
      }
      throw error
    }
  }
}

export const name = 'tool-rewind'

/** Config: the model-facing tool name plus summarization settings. */
export interface Config {
  toolName: string
  maxTokens: number
  maxSummarizationRetries: number
  summarizationProvider?: string
  summarizationModel?: string
  reportLanguage: 'en' | 'zh'
}

export const Config: z<Config> = z.object({
  toolName: z.string().default('rewind'),
  maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(1024),
  maxSummarizationRetries: z.number().step(1).min(0).max(8).default(2),
  summarizationProvider: z.string(),
  summarizationModel: z.string(),
  reportLanguage: z.union(['en', 'zh'] as const).default('en'),
})

/**
 * Compose the entry plugin: install the turn-ending fold guard, provide the
 * fold service, then mount the tool child plugin (which injects the service
 * it delegates to). The service activation is awaited — its `llm` inject
 * resolves asynchronously.
 */
export async function apply(ctx: Context, config: Config): Promise<void> {
  installRewindGuard(ctx)
  await ctx.plugin(RewindService, config)
  await ctx.plugin(toolPlugin, config)
}
