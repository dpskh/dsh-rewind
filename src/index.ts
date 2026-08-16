/**
 * Exploration-fold plugin: registers `ctx.rewind` (a service that folds
 * everything since the most recent `checkpoint/mark` into an auto-generated
 * report) and the model-facing `rewind` tool over it. The fold shadows the
 * exploration's surface range with the report — the same `surfaceOp:
 * { op: 'replace' }` mechanism the compaction seam uses — so subsequent
 * requests no longer carry the exploration's intermediate steps, while the
 * durable log keeps the full exploration for audit.
 *
 * Marks are owned by `@dpskh/tool-checkpoint` and consumed exclusively
 * through `ctx.checkpoint` (`latestMark` / `hasActive` / `completeFold`);
 * this plugin writes no session events of its own — the fold commits only
 * the core-vocabulary `user/message` replacement, so the durable log stays
 * readable by any harness. Consumes `ctx.llm` for the report and the
 * compaction seam's exported pair-balance helpers; never modifies upstream
 * packages.
 *
 * @module @dpskh/tool-rewind
 */

import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { SessionId } from '@deepseek-ai/dsh-session'
import type { CheckpointService } from '@dpskh/tool-checkpoint'
import * as toolPlugin from './tool.ts'

import { REWIND_REPORT_SOURCE } from './brand.ts'
import { installRewindGuard } from './guard.ts'
import {
  RewindError,
  selectFoldRegion,
  assertRegionUnchanged,
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
  /** Id of the `checkpoint/mark` the fold started after. */
  markId: number
  /** Number of surface nodes folded into the report. */
  foldedNodes: number
  /** Inclusive first folded surface-node seq. */
  start: number
  /** Inclusive last folded surface-node seq. */
  end: number
  /** Model-visible text chars the folded region contributed (the shrink yardstick). */
  foldedChars: number
  /** Report chars that replaced the region. */
  reportChars: number
}

/**
 * The exploration-fold service, registered as `ctx.rewind`. One instance per
 * context; the verb runs the full fold transaction: find the latest mark
 * through `ctx.checkpoint`, select the balanced region, summarize it over
 * `ctx.llm`, then commit the replacement in one synchronous block and stamp
 * the mark folded. Failures write nothing — the region stays intact and the
 * fold is retryable. A per-session in-process lock rejects concurrent folds
 * (the old log-bracket lock no longer exists, since no fold events are
 * written).
 */
export class RewindService extends Service {
  static inject = ['llm', 'checkpoint']

  private readonly maxTokens: number
  private readonly maxSummarizationRetries: number
  private readonly summarizationProvider: string | undefined
  private readonly summarizationModel: string | undefined
  private readonly reportLanguage: 'en' | 'zh'
  /** Per-session fold lock: one in-flight fold per session. */
  private readonly foldLocks = new Set<SessionId>()

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
    const checkpoint: CheckpointService = this.ctx.checkpoint
    const mark = await checkpoint.latestMark(session.id)
    if (mark === undefined) {
      throw new RewindError('NO_CHECKPOINT', 'rewind: no checkpoint mark found for this session; call `checkpoint` before exploring')
    }
    if (this.foldLocks.has(session.id)) {
      throw new RewindError(
        'FOLD_IN_PROGRESS',
        'rewind: another fold is already in progress for this session',
      )
    }
    this.foldLocks.add(session.id)
    try {
      const region = selectFoldRegion(session, mark.logLength)
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
      assertRegionUnchanged(session, mark.logLength, region)
      const foldedText = regionTextLength(input.messages)
      if (foldedText > 0 && summary.report.length >= foldedText) {
        throw new RewindError(
          'SHRINK',
          `rewind summary (${summary.report.length} chars) is not smaller than the folded region (${foldedText} chars over ${region.shadowedSeqs.length} nodes)`,
        )
      }
      // Synchronous commit: the replacement lands, then the mark is stamped
      // folded. Nothing between the append and the completeFold may throw
      // after the append succeeded.
      session.append('user/message', createUserMessage({
        content: [{ type: 'text', text: summary.report }],
        source: REWIND_REPORT_SOURCE,
      }), {
        surfaceOp: { op: 'replace', start: region.start, end: region.end },
        sourceEventSeqs: [...region.shadowedSeqs],
      })
      await checkpoint.completeFold(session.id, mark.id)
      return {
        markId: mark.id,
        foldedNodes: region.shadowedSeqs.length,
        start: region.start,
        end: region.end,
        foldedChars: foldedText,
        reportChars: summary.report.length,
      }
    } finally {
      this.foldLocks.delete(session.id)
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
 * it delegates to). The service activation is awaited — its `llm` and
 * `checkpoint` injects resolve asynchronously.
 */
export async function apply(ctx: Context, config: Config): Promise<void> {
  installRewindGuard(ctx)
  await ctx.plugin(RewindService, config)
  await ctx.plugin(toolPlugin, config)
}
