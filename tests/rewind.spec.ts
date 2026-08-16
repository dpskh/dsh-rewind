import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import Storage from '@deepseek-ai/dsh-storage'
import { SqliteStorageBackend } from '@deepseek-ai/dsh-storage-sqlite'
import { DomainFacility } from '@deepseek-ai/dsh-storage-domain'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRegistry from '@deepseek-ai/dsh-tools'
import LlmService, { CallId, LlmAdapter, ReasoningEffortId, createMessage, createToolResultMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, LlmResolvedModelInfo, Message, StreamChunk } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import type { Agent } from '@deepseek-ai/dsh-agent'
import * as checkpointTool from '@dpskh/tool-checkpoint'

import { isRewindReportSource } from '../src/brand.ts'
import { REWIND_GUARD_SOURCE, REWIND_GUARD_TEXT } from '../src/guard.ts'
import { RewindError, assertRegionUnchanged, selectFoldRegion } from '../src/region.ts'
import { buildSummarizationInput, summarizeRegion } from '../src/summarize.ts'
import { RewindService } from '../src/index.ts'
import * as tool from '../src/index.ts'
import type { Config } from '../src/index.ts'

const SIGNAL = new AbortController().signal
const MODEL = 'test'

/** A parent Agent backed by a real Session — the service reads `agent.session`. */
function agentWithSession(session: Session, options: { provider?: string; model?: string } = {}): Agent {
  return {
    id: session.id,
    session,
    options: options.provider === undefined ? {} : { provider: options.provider, model: options.model },
  } as unknown as Agent
}

/** One assistant message hosting a single tool call. */
function toolCallMessage(callId: string, name: string) {
  return createMessage({
    role: 'assistant',
    content: [{ type: 'tool-call', id: CallId(callId), name, arguments: '{}' }],
    source: { kind: 'model', provider: MODEL, model: MODEL },
  })
}

/**
 * A session in the middle of an exploration: a checkpoint mark sits after
 * the user's request, the exploration is one bash call with its result, and
 * the current step's assistant message carries the rewind call itself.
 * Surface nodes: [user/message, assistant(bash), tool/result, assistant(rewind)].
 *
 * The mark itself is not a session event — it lives in the checkpoint
 * storage domain, recorded by the caller (typically through
 * `ctx.checkpoint.mark`) at the moment the session reaches the marked
 * position. `EXPLORATION_MARK_LOG_LENGTH` is the log length at that moment
 * (4: seqs 0-3 are present), i.e. the fold anchor: every surface node with
 * `seq >= 4` came after the mark.
 */
const EXPLORATION_MARK_LOG_LENGTH = 4

function explorationSession(): Session {
  const session = Session.create(SessionId('explore'))
  session.append('turn/start', { turn: 1 })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Find the cause' }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn: 1, step: 1 })
  session.append('request/header', { header: { config: { provider: MODEL, model: MODEL } }, reason: 'initial' })
  session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'bash') }, { surfaceOp: 'append' })
  session.append('tool/result', {
    turn: 1,
    step: 1,
    message: createToolResultMessage({ callId: CallId('c1'), content: [{ type: 'text', text: 'exploration output line' }], isError: false }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn: 1, step: 1 })
  session.append('step/start', { turn: 1, step: 2 })
  session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
  return session
}

/**
 * Build the exploration session with a REAL mark recorded through the
 * checkpoint service at the marked position: the mark lands after the
 * user request and request header, before the exploration's bash call —
 * exactly where the `checkpoint` tool runs in a live loop.
 */
async function markedExplorationSession(ctx: Context, objective?: string): Promise<Session> {
  const session = Session.create(SessionId('explore'))
  session.append('turn/start', { turn: 1 })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Find the cause' }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn: 1, step: 1 })
  session.append('request/header', { header: { config: { provider: MODEL, model: MODEL } }, reason: 'initial' })
  await ctx.checkpoint.mark(session, objective)
  session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'bash') }, { surfaceOp: 'append' })
  session.append('tool/result', {
    turn: 1,
    step: 1,
    message: createToolResultMessage({ callId: CallId('c1'), content: [{ type: 'text', text: 'exploration output line' }], isError: false }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn: 1, step: 1 })
  session.append('step/start', { turn: 1, step: 2 })
  session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
  return session
}

/** A scripted llm adapter: yields fixed chunks and captures every request. */
class ScriptedAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = []

  constructor(
    private readonly chunks: StreamChunk[],
    private readonly reasoning?: LlmResolvedModelInfo['reasoning'],
  ) {
    super()
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      ...this.reasoning === undefined ? {} : { reasoning: this.reasoning },
    })
  }

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    for (const chunk of this.chunks) yield chunk
  }
}

/** A scripted adapter with one chunk set per stream call (retry tests). */
class SequenceAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = []

  constructor(private readonly sequences: StreamChunk[][]) {
    super()
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({ provider, id: model, name: model })
  }

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    const chunks = this.sequences[this.requests.length - 1]
    if (chunks !== undefined) {
      for (const chunk of chunks) yield chunk
    }
  }
}

function textChunks(text: string): StreamChunk[] {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text },
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

/** Mount the storage stack manually (the workspace-spec pattern). */
async function mountStorage(ctx: Context): Promise<void> {
  await ctx.plugin(Storage)
  const backend = new SqliteStorageBackend({ path: ':memory:', journalMode: 'wal' })
  ctx.storage.backend.register('sqlite', backend)
  const facility = new DomainFacility(ctx, { backend: 'sqlite' })
  ctx.storage.mount('domain', facility)
  ctx.provide('storageDomain', facility)
}

async function setup(
  chunks: StreamChunk[],
  config: Partial<Config> = {},
): Promise<{ ctx: Context; adapter: ScriptedAdapter }> {
  const ctx = new Context()
  await mountStorage(ctx)
  await ctx.plugin(LlmService)
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(checkpointTool)
  const adapter = new ScriptedAdapter(chunks)
  ctx.llm.registerAdapter([MODEL, 'other'], adapter)
  // Partial is fine at the mount boundary: the plugin Config schema applies
  // defaults for absent fields (exactOptionalPropertyTypes forbids the spread).
  await ctx.plugin(tool, config as Config)
  return { ctx, adapter }
}

/** Record a mark for the session through the real checkpoint service. */
async function mark(ctx: Context, session: Session, objective?: string): Promise<void> {
  await ctx.checkpoint.mark(session, objective)
}

describe('region selection', () => {
  it('selects the balanced span after the mark anchor, excluding the rewind call', () => {
    const session = explorationSession()
    const region = selectFoldRegion(session, EXPLORATION_MARK_LOG_LENGTH)
    expect(region.start).toBe(4)
    expect(region.end).toBe(5)
    expect(region.shadowedSeqs).toEqual([4, 5])
  })

  it('rejects a fold with nothing after the anchor', () => {
    const bare = Session.create(SessionId('bare'))
    expect(() => selectFoldRegion(bare, 0)).toThrow(RewindError)
    expect(() => selectFoldRegion(bare, 0)).toThrow(/no surface nodes after the checkpoint/)

    // An anchor at the end of the log: nothing after it.
    const session = explorationSession()
    expect(() => selectFoldRegion(session, session.events.length)).toThrow(/no surface nodes after the checkpoint/)
  })

  it('rejects a fold with nothing between the mark and the rewind call', () => {
    // Build a fresh session where the mark lands right before the rewind call.
    const tight = Session.create(SessionId('tight'))
    tight.append('turn/start', { turn: 1 })
    tight.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'Find the cause' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    const anchor = tight.events.length
    tight.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    expect(() => selectFoldRegion(tight, anchor)).toThrow(/no surface nodes between the checkpoint/)
  })

  it('rejects an unbalanced region whose open tool call never resolves', () => {
    const session = Session.create(SessionId('unbalanced'))
    session.append('turn/start', { turn: 1 })
    const anchor = session.events.length
    session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'bash') }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    expect(() => selectFoldRegion(session, anchor)).toThrow(/balanced boundary/)
  })

  it('folds an exploration whose checkpoint call spans the mark', () => {
    // The real checkpoint tool runs between its own assistant tool-call and
    // tool/result, so the mark lands inside that pair: the orphaned result
    // is skipped and the fold starts at the exploration that follows.
    const session = Session.create(SessionId('midcall'))
    session.append('turn/start', { turn: 1 })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'Find the cause' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'checkpoint') }, { surfaceOp: 'append' })
    const anchor = session.events.length
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [{ type: 'text', text: '{"id":1}' }], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'bash') }, { surfaceOp: 'append' })
    session.append('tool/result', { turn: 1, step: 2, message: createToolResultMessage({ callId: CallId('c2'), content: [{ type: 'text', text: 'exploration output line' }], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 3, message: toolCallMessage('c3', 'rewind') }, { surfaceOp: 'append' })
    const region = selectFoldRegion(session, anchor)
    expect(region.shadowedSeqs).toEqual([4, 5])
    // The checkpoint call's own pair stays outside the fold and model-visible.
    expect(region.start).toBe(4)
    expect(region.end).toBe(5)
  })

  it('skips an orphaned result and folds nothing when only the rewind call follows', () => {
    const session = Session.create(SessionId('splitstart'))
    session.append('turn/start', { turn: 1 })
    // The assistant requests TWO calls; only the first resolves before the
    // second mark, so the region after it starts at an unmatched result.
    session.append('assistant/message', {
      turn: 1, step: 1,
      message: createMessage({
        role: 'assistant',
        content: [
          { type: 'tool-call', id: CallId('c1'), name: 'bash', arguments: '{}' },
          { type: 'tool-call', id: CallId('c2'), name: 'bash', arguments: '{}' },
        ],
        source: { kind: 'model', provider: MODEL, model: MODEL },
      }),
    }, { surfaceOp: 'append' })
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [], isError: false }) }, { surfaceOp: 'append' })
    const anchor = session.events.length
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c2'), content: [], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c3', 'rewind') }, { surfaceOp: 'append' })
    expect(() => selectFoldRegion(session, anchor)).toThrow(/no surface nodes between the checkpoint/)
  })

  it('folds an exploration issued in parallel with the checkpoint call', () => {
    // The model emits the checkpoint and an exploration call in ONE assistant
    // message (parallel tool calls); the mark lands between the checkpoint
    // call and its result, and the exploration result follows. Selection must
    // skip only the checkpoint's orphaned result and fold the exploration's
    // own nodes — never report an empty region.
    const session = Session.create(SessionId('parallel'))
    session.append('turn/start', { turn: 1 })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'Explore it' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('assistant/message', {
      turn: 1, step: 1,
      message: createMessage({
        role: 'assistant',
        content: [
          { type: 'tool-call', id: CallId('c1'), name: 'checkpoint', arguments: '{}' },
          { type: 'tool-call', id: CallId('c2'), name: 'bash', arguments: '{"command":"probe"}' },
        ],
        source: { kind: 'model', provider: MODEL, model: MODEL },
      }),
    }, { surfaceOp: 'append' })
    const anchor = session.events.length
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [{ type: 'text', text: 'marked' }], isError: false }) }, { surfaceOp: 'append' })
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c2'), content: [{ type: 'text', text: 'probe output' }], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c3', 'rewind') }, { surfaceOp: 'append' })
    const region = selectFoldRegion(session, anchor)
    // The checkpoint call's orphaned result (seq 3) stays visible; the
    // exploration result (seq 4) is the fold.
    expect(region.shadowedSeqs).toEqual([4])
  })

  it('rejects when an orphaned result is the only node after the mark', () => {
    const session = Session.create(SessionId('lone-result'))
    session.append('turn/start', { turn: 1 })
    session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'checkpoint') }, { surfaceOp: 'append' })
    const anchor = session.events.length
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [], isError: false }) }, { surfaceOp: 'append' })
    expect(() => selectFoldRegion(session, anchor)).toThrow(/no surface nodes between the checkpoint/)
  })

  it('folds to the surface tail when no assistant message exists yet', () => {
    const session = Session.create(SessionId('noassistant'))
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'a' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    const region = selectFoldRegion(session, 0)
    expect(region.shadowedSeqs).toEqual([0])
  })

  it('asserts the region stability', () => {
    const session = explorationSession()
    const region = selectFoldRegion(session, EXPLORATION_MARK_LOG_LENGTH)
    expect(() => { assertRegionUnchanged(session, EXPLORATION_MARK_LOG_LENGTH, region) }).not.toThrow()
    expect(() => { assertRegionUnchanged(session, EXPLORATION_MARK_LOG_LENGTH, { ...region, end: region.end + 1 }) })
      .toThrow(/changed during summarization/)
  })
})

describe('rewind service', () => {
  it('resolves service config with defaults', () => {
    const read = (service: RewindService): { maxTokens: number; retries: number; language: string } => ({
      maxTokens: (service as unknown as { maxTokens: number }).maxTokens,
      retries: (service as unknown as { maxSummarizationRetries: number }).maxSummarizationRetries,
      language: (service as unknown as { reportLanguage: string }).reportLanguage,
    })
    expect(read(new RewindService(new Context(), {}))).toEqual({ maxTokens: 1024, retries: 2, language: 'en' })
    expect(read(new RewindService(new Context(), { maxTokens: 512, maxSummarizationRetries: 0, reportLanguage: 'zh' })))
      .toEqual({ maxTokens: 512, retries: 0, language: 'zh' })
  })

  it('folds the exploration into an auto-generated report', async () => {
    const { ctx } = await setup(textChunks('## Findings\n- X'))
    const session = await markedExplorationSession(ctx, 'explore the failure')
    const result = await ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL)
    // The region is the bash call (arguments '{}' = 2 chars) plus its result
    // ('exploration output line' = 23 chars): 25 model-visible chars total.
    expect(result).toEqual({ markId: 1, foldedNodes: 2, start: 4, end: 5, foldedChars: 25, reportChars: 15 })

    // No plugin events enter the session log: only the core-vocabulary
    // replacement message is appended.
    expect(session.events.some(e => e.type.startsWith('checkpoint/'))).toBe(false)
    const stored = await ctx.checkpoint.latestMark(session.id)
    expect(stored?.foldedAt).not.toBeNull()

    const report = session.events.find(e => e.type === 'user/message'
      && e.data.source.kind === 'plugin' && e.data.source.plugin === 'rewind') as SessionEvent<'user/message'> | undefined
    expect(report?.surfaceOp).toEqual({ op: 'replace', start: 4, end: 5 })
    const messages = session.deriveMessages()
    expect(messages.map(m => m.content)).toEqual([
      [{ type: 'text', text: 'Find the cause' }],
      [{ type: 'text', text: '## Findings\n- X' }],
      [{ type: 'tool-call', id: CallId('c2'), name: 'rewind', arguments: '{}' }],
    ])
  })

  it('completes the fold and stamps the mark folded', async () => {
    const chunks: StreamChunk[] = [
      ...textChunks('report').slice(0, 3),
      { type: 'usage', usage: { inputTokens: 1, outputTokens: 2 } },
      textChunks('report')[3]!,
    ]
    const { ctx } = await setup(chunks)
    const session = await markedExplorationSession(ctx)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(true)
    await ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(false)
    const stored = await ctx.checkpoint.latestMark(session.id)
    expect(stored?.foldedAt).not.toBeNull()
  })

  it('routes the summarization request to the routed or configured model', async () => {
    const { ctx, adapter } = await setup(textChunks('report'))
    const session = await markedExplorationSession(ctx)
    await ctx.rewind.rewind(agentWithSession(session), SIGNAL)
    expect(adapter.requests).toHaveLength(1)
    expect(adapter.requests[0]!.provider).toBe(MODEL)
    expect(adapter.requests[0]!.model).toBe(MODEL)
    expect(adapter.requests[0]!.maxTokens).toBe(1024)
    // The instruction rides as the final user message after the region.
    const last = adapter.requests[0]!.messages.at(-1)!
    expect(last.role).toBe('user')
    expect(JSON.stringify(last.content)).toContain('exploration-report engine')
    // The region messages precede it.
    expect(adapter.requests[0]!.messages).toHaveLength(3)
  })

  it('prefers configured summarization fields and honors the zh report language', async () => {
    const { ctx, adapter } = await setup(textChunks('报告'), {
      summarizationProvider: 'other',
      summarizationModel: 'other-model',
      reportLanguage: 'zh',
      maxTokens: 256,
    })
    const session = await markedExplorationSession(ctx)
    await ctx.rewind.rewind(agentWithSession(session), SIGNAL)
    expect(adapter.requests[0]!.provider).toBe('other')
    expect(adapter.requests[0]!.maxTokens).toBe(256)
    const last = adapter.requests[0]!.messages.at(-1)!
    expect(JSON.stringify(last.content)).toContain('探索报告引擎')
  })

  it('rejects a fold without a checkpoint mark', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = Session.create(SessionId('bare'))
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL)).rejects.toThrow(/no checkpoint mark/)
  })

  it('rejects a fold while another fold is in progress', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = await markedExplorationSession(ctx)
    ;(ctx.rewind as unknown as { foldLocks: Set<SessionId> }).foldLocks.add(session.id)
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL)).rejects.toThrow(/already in progress/)
  })

  it('rejects a summary that is not smaller than the folded region', async () => {
    const { ctx } = await setup(textChunks('x'.repeat(200)))
    const session = Session.create(SessionId('shrink'))
    session.append('turn/start', { turn: 1 })
    await mark(ctx, session)
    session.append('assistant/message', {
      turn: 1, step: 1,
      message: createMessage({
        role: 'assistant',
        content: [{ type: 'text', text: 'short analysis' }],
        source: { kind: 'model', provider: MODEL, model: MODEL },
      }),
    }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    await expect(ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL)).rejects.toThrow(/not smaller/)
  })

  it('skips empty-content assistant nodes when building the summarization input', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = Session.create(SessionId('emptyassistant'))
    session.append('turn/start', { turn: 1 })
    await mark(ctx, session)
    // An empty-content assistant/message derives to no LLM message (it only
    // hosts a max-tokens step's usage), so the summarization input skips it.
    session.append('assistant/message', {
      turn: 1, step: 1,
      message: createMessage({ role: 'assistant', content: [], source: { kind: 'model', provider: MODEL, model: MODEL } }),
    }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    await expect(ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL))
      .resolves.toMatchObject({ foldedNodes: 1 })
  })

  it('skips the shrink check when the folded region carries no text', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = Session.create(SessionId('toolonly'))
    session.append('turn/start', { turn: 1 })
    await mark(ctx, session)
    // A tool call with empty arguments and an empty result contributes no
    // model-visible text, so the region's char yardstick is zero and the
    // shrink check has nothing to compare (no false SHRINK rejection).
    session.append('assistant/message', {
      turn: 1, step: 1,
      message: createMessage({
        role: 'assistant',
        content: [{ type: 'tool-call', id: CallId('c1'), name: 'bash', arguments: '' }],
        source: { kind: 'model', provider: MODEL, model: MODEL },
      }),
    }, { surfaceOp: 'append' })
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    await expect(ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL))
      .resolves.toMatchObject({ foldedNodes: 2, foldedChars: 0, reportChars: 6 })
  })

  it('keeps the mark active and writes nothing when summarization fails', async () => {
    const { ctx } = await setup([{ type: 'finish', reason: { kind: 'stop' } }], { maxSummarizationRetries: 0 })
    const session = await markedExplorationSession(ctx)
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL)).rejects.toThrow(/no text report/)
    expect(session.events.some(e => e.type.startsWith('checkpoint/'))).toBe(false)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(true)
  })

  it('retries an empty completion and succeeds on a later attempt', async () => {
    const adapter = new SequenceAdapter([
      [{ type: 'finish', reason: { kind: 'stop' } }],
      textChunks('report'),
    ])
    const ctx = new Context()
    await mountStorage(ctx)
    await ctx.plugin(LlmService)
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(checkpointTool)
    ctx.llm.registerAdapter([MODEL], adapter)
    // Partial is fine at the mount boundary: the plugin Config schema applies
    // defaults for absent fields (exactOptionalPropertyTypes forbids the spread).
    await ctx.plugin(tool, {} as Config)
    const session = await markedExplorationSession(ctx)
    await expect(ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL))
      .resolves.toMatchObject({ foldedNodes: 2 })
    expect(adapter.requests).toHaveLength(2)
  })

  it('exhausts the retry budget and reports finish/usage diagnostics', async () => {
    const { ctx, adapter } = await setup(
      [{ type: 'usage', usage: { inputTokens: 1, outputTokens: 1024, reasoningTokens: 1024 } }, { type: 'finish', reason: { kind: 'max-tokens' } }],
      { maxSummarizationRetries: 2 },
    )
    const session = await markedExplorationSession(ctx)
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL))
      .rejects.toThrow(/no text report content after 2 retries \(finish: max-tokens, outputTokens: 1024, reasoningTokens: 1024\)/)
    expect(session.events.some(e => e.type.startsWith('checkpoint/'))).toBe(false)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(true)
    expect(adapter.requests).toHaveLength(3)
  })

  it('reports output tokens alone when the exhausted call reports no reasoning usage', async () => {
    const { ctx } = await setup(
      [{ type: 'usage', usage: { inputTokens: 1, outputTokens: 1024 } }, { type: 'finish', reason: { kind: 'max-tokens' } }],
      { maxSummarizationRetries: 0 },
    )
    const session = await markedExplorationSession(ctx)
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL))
      .rejects.toThrow(/no text report content after 0 retries \(finish: max-tokens, outputTokens: 1024\)/)
  })

  it('disables thinking for the report call when the route exposes an off effort', async () => {
    const adapter = new ScriptedAdapter(textChunks('report'), {
      efforts: [
        { id: ReasoningEffortId('off'), name: 'Off' },
        { id: ReasoningEffortId('high'), name: 'High' },
      ],
      defaultEffort: ReasoningEffortId('high'),
    })
    const ctx = new Context()
    await mountStorage(ctx)
    await ctx.plugin(LlmService)
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(checkpointTool)
    ctx.llm.registerAdapter([MODEL], adapter)
    // Partial is fine at the mount boundary: the plugin Config schema applies
    // defaults for absent fields (exactOptionalPropertyTypes forbids the spread).
    await ctx.plugin(tool, {} as Config)
    const session = await markedExplorationSession(ctx)
    await ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL)
    expect(adapter.requests[0]!.reasoningEffort).toBe('off')
  })

  it('keeps the route default effort when no off effort is advertised', async () => {
    const { ctx, adapter } = await setup(textChunks('report'))
    const session = await markedExplorationSession(ctx)
    await ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL)
    expect(adapter.requests[0]!.reasoningEffort).toBeUndefined()
  })

  it('keeps the mark active and writes nothing when the summarization stream errors', async () => {
    const { ctx } = await setup([{ type: 'finish', reason: { kind: 'error', failure: { message: 'boom', code: 'E' } } }])
    const session = await markedExplorationSession(ctx)
    await expect(ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL))
      .rejects.toThrow(/stream ended with error/)
    expect(session.events.some(e => e.type.startsWith('checkpoint/'))).toBe(false)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(true)
  })

  it('propagates a cancelled signal and writes nothing', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = await markedExplorationSession(ctx)
    const aborted = new AbortController()
    aborted.abort()
    await expect(ctx.rewind.rewind(agentWithSession(session), aborted.signal)).rejects.toBeDefined()
    expect(session.events.some(e => e.type.startsWith('checkpoint/'))).toBe(false)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(true)
  })

  it('rejects when no provider/model can be resolved for summarization', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = Session.create(SessionId('nohost'))
    await mark(ctx, session)
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'explore' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 0, step: 0, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    // No request/header and no agent options: resolution must fail loud.
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL)).rejects.toThrow(/no provider\/model available/)
  })
})

describe('summarizeRegion', () => {
  it('passes the session system and tools through for prefix alignment', async () => {
    const { ctx, adapter } = await setup(textChunks('report'))
    const session = Session.create(SessionId('sys'))
    session.append('request/header', {
      header: {
        config: { provider: MODEL, model: MODEL },
        system: 'sys-text',
        tools: [{ name: 't', description: 'd', parameters: { type: 'object', properties: {} } }],
      },
      reason: 'initial',
    })
    const summary = await summarizeRegion(
      ctx,
      buildSummarizationInput(session, { start: 0, end: -1, shadowedSeqs: [] }),
      agentWithSession(session, { provider: MODEL, model: MODEL }),
      { maxTokens: 64, maxSummarizationRetries: 0, reportLanguage: 'en' },
    )
    expect(summary.report).toBe('report')
    expect(adapter.requests[0]!.system).toBe('sys-text')
    expect(adapter.requests[0]!.tools).toEqual([{ name: 't', description: 'd', parameters: { type: 'object', properties: {} } }])
    // Direct call without a signal covers the signal-less stream path.
    expect(adapter.requests[0]!.signal).toBeUndefined()
  })

  it('falls back to agent options when no routed header exists', async () => {
    const { ctx, adapter } = await setup(textChunks('report'))
    const session = Session.create(SessionId('noheader'))
    await summarizeRegion(
      ctx,
      { messages: [] },
      agentWithSession(session, { provider: MODEL, model: MODEL }),
      { maxTokens: 64, maxSummarizationRetries: 0, reportLanguage: 'en' },
    )
    expect(adapter.requests[0]!.provider).toBe(MODEL)
  })

  it.each([
    [{ provider: MODEL }, /no provider\/model available/],
    [{ provider: '' }, /no provider\/model available/],
    [{ provider: MODEL, model: '' }, /no provider\/model available/],
  ])('rejects partial agent options %j', async (options, message) => {
    const { ctx } = await setup(textChunks('report'))
    const session = Session.create(SessionId('partial'))
    await expect(summarizeRegion(
      ctx,
      { messages: [] },
      agentWithSession(session, options),
      { maxTokens: 64, maxSummarizationRetries: 0, reportLanguage: 'en' },
    )).rejects.toThrow(message)
  })
})

describe('rewind tool', () => {
  async function setupTools(chunks: StreamChunk[]): Promise<Context> {
    const ctx = new Context()
    await mountStorage(ctx)
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRegistry)
    await ctx.plugin(LlmService)
    await ctx.plugin(checkpointTool)
    const adapter = new ScriptedAdapter(chunks)
    ctx.llm.registerAdapter([MODEL], adapter)
    await ctx.plugin(tool)
    return ctx
  }

  let callCounter = 0
  function callRewind(ctx: Context, agent: Agent | undefined) {
    return ctx.tools.execute({
      signal: SIGNAL,
      callId: CallId(`call-${++callCounter}`),
      name: 'rewind',
      arguments: {},
      ...agent === undefined ? {} : { agent },
    })
  }

  it('registers a rewind tool and folds through the real registry', async () => {
    const ctx = await setupTools(textChunks('## Findings\n- done'))
    const schema = ctx.tools.schemas().find(s => s.name === 'rewind')
    expect(schema).toBeDefined()
    const session = await markedExplorationSession(ctx)
    const result = await callRewind(ctx, agentWithSession(session, { provider: MODEL, model: MODEL }))
    const text = result.content.filter(b => b.type === 'text').map(b => b.text).join('')
    expect(text).toMatch(/Folded 2 surface nodes/)
    expect(session.events.some(e => e.type.startsWith('checkpoint/'))).toBe(false)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(false)
  })

  it('presents the call and the result with a stable generic card', async () => {
    const ctx = await setupTools(textChunks('report'))
    const def = ctx.tools.get('rewind')!
    expect(def.presentCall?.({})).toEqual({ card: 'generic', title: 'Rewind', kind: 'other' })
    const rendered = def.output.render({}, { markId: 4, foldedNodes: 2, start: 5, end: 6, foldedChars: 25, reportChars: 15 })
    expect(rendered).toEqual([{ type: 'text', text: 'Folded 2 surface nodes (seq 5..6, 25 chars) into an auto-generated report (15 chars).' }])
    const single = def.output.render({}, { markId: 1, foldedNodes: 1, start: 2, end: 2, foldedChars: 0, reportChars: 0 })
    expect(single).toEqual([{ type: 'text', text: 'Folded 1 surface node (seq 2..2, 0 chars) into an auto-generated report (0 chars).' }])
  })

  it('registers the fold-discipline prompt section', async () => {
    const ctx = await setupTools(textChunks('report'))
    const assembly = await ctx.systemPrompt.assemble()
    const section = assembly.sections.find(s => s.name === 'tool:rewind')
    expect(section?.text).toContain('MUST call `rewind` as soon as a marked exploration is done')
    expect(section?.text).toContain('Without a preceding `checkpoint` mark it errors')
  })

  it('rejects a call without an owning agent session', async () => {
    const ctx = await setupTools(textChunks('report'))
    const result = await callRewind(ctx, undefined)
    expect(result.isError).toBe(true)
    expect(result.error?.message).toMatch(/owning agent session/)
  })
})

describe('report source marker', () => {
  it('recognizes the rewind report source and nothing else', () => {
    expect(isRewindReportSource({ kind: 'plugin', plugin: 'rewind' })).toBe(true)
    expect(isRewindReportSource({ kind: 'user' })).toBe(false)
    expect(isRewindReportSource({ kind: 'plugin', plugin: 'compact' })).toBe(false)
  })
})

describe('turn guard', () => {
  async function setupGuard(): Promise<Context> {
    const ctx = new Context()
    await mountStorage(ctx)
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRegistry)
    await ctx.plugin(LlmService)
    await ctx.plugin(checkpointTool)
    const adapter = new ScriptedAdapter(textChunks('report'))
    ctx.llm.registerAdapter([MODEL], adapter)
    await ctx.plugin(tool)
    return ctx
  }

  it('reports an active checkpoint from the plugin storage', async () => {
    const ctx = await setupGuard()
    const session = Session.create(SessionId('guard-state'))
    session.append('turn/start', { turn: 1 })
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(false)
    const { id } = await ctx.checkpoint.mark(session)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(true)
    // Completing a fold for a different (unknown) mark id keeps it active.
    await ctx.checkpoint.completeFold(session.id, id - 1)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(true)
    // Completing the latest mark closes it.
    await ctx.checkpoint.completeFold(session.id, id)
    expect(await ctx.checkpoint.hasActive(session.id)).toBe(false)
  })

  it('injects the warning once per turn while a checkpoint is active', async () => {
    const ctx = await setupGuard()
    const session = Session.create(SessionId('guard-inject'))
    session.append('turn/start', { turn: 1 })
    const { id } = await ctx.checkpoint.mark(session)
    const inject = vi.fn()
    const agent = { id: SessionId('guard-inject'), session, inject } as unknown as Agent

    ctx.emit('agent/turn-stopping', { agent, turn: 1, signal: SIGNAL })
    await vi.waitFor(() => {
      expect(inject).toHaveBeenCalledTimes(1)
    })
    const message = inject.mock.calls[0]![0] as Message
    expect(message.content).toEqual([{ type: 'text', text: REWIND_GUARD_TEXT }])
    expect(message.source).toEqual(REWIND_GUARD_SOURCE)

    // A second stop attempt in the same turn is not warned again — a rewind
    // that legitimately fails must not trap the turn in a warning loop.
    ctx.emit('agent/turn-stopping', { agent, turn: 1, signal: SIGNAL })
    await vi.waitFor(() => {
      expect(inject).toHaveBeenCalledTimes(1)
    })

    // After the mark is folded, a later turn stops quietly.
    await ctx.checkpoint.completeFold(session.id, id)
    ctx.emit('agent/turn-stopping', { agent, turn: 2, signal: SIGNAL })
    await vi.waitFor(() => {
      expect(inject).toHaveBeenCalledTimes(1)
    })
  })
})
