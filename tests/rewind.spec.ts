import { describe, expect, it } from 'vitest'
import { Context } from 'cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRegistry from '@deepseek-ai/dsh-tools'
import LlmService, { CallId, LlmAdapter, createMessage, createToolResultMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, LlmResolvedModelInfo, StreamChunk } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import type { Agent } from '@deepseek-ai/dsh-agent'

import { isRewindReportSource } from '../src/brand.ts'
import { RewindError, assertNoActiveFold, assertRegionUnchanged, findLatestMark, openTurnOf, selectFoldRegion } from '../src/region.ts'
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
 * A session in the middle of an exploration: the checkpoint mark sits after
 * the user's request, the exploration is one bash call with its result, and
 * the current step's assistant message carries the rewind call itself.
 * Surface nodes: [user/message, assistant(bash), tool/result, assistant(rewind)].
 */
function explorationSession(): Session {
  const session = Session.create(SessionId('explore'))
  session.append('turn/start', { turn: 1 })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Find the cause' }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn: 1, step: 1 })
  session.append('request/header', { header: { config: { provider: MODEL, model: MODEL } }, reason: 'initial' })
  session.append('checkpoint/mark', { turn: 1, objective: 'explore the failure' })
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

  constructor(private readonly chunks: StreamChunk[]) {
    super()
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({ provider, id: model, name: model })
  }

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    for (const chunk of this.chunks) yield chunk
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

async function setup(
  chunks: StreamChunk[],
  config: Partial<Config> = {},
): Promise<{ ctx: Context; adapter: ScriptedAdapter }> {
  const ctx = new Context()
  await ctx.plugin(LlmService)
  await ctx.plugin(SystemPrompt)
  const adapter = new ScriptedAdapter(chunks)
  ctx.llm.registerAdapter([MODEL, 'other'], adapter)
  // Partial is fine at the mount boundary: the plugin Config schema applies
  // defaults for absent fields (exactOptionalPropertyTypes forbids the spread).
  await ctx.plugin(tool, config as Config)
  return { ctx, adapter }
}

describe('region selection', () => {
  it('finds the latest checkpoint mark', () => {
    const session = explorationSession()
    const marks = session.events.filter(e => e.type === 'checkpoint/mark')
    expect(marks).toHaveLength(1)
    expect(findLatestMark(session.events)?.seq).toBe(marks[0]!.seq)
    expect(findLatestMark(Session.create(SessionId('empty')).events)).toBeUndefined()
  })

  it('selects the balanced span after the mark, excluding the rewind call', () => {
    const session = explorationSession()
    const mark = findLatestMark(session.events)!
    const region = selectFoldRegion(session, mark.seq)
    expect(region.start).toBe(5)
    expect(region.end).toBe(6)
    expect(region.shadowedSeqs).toEqual([5, 6])
  })

  it('rejects a fold with no mark or nothing after it', () => {
    const bare = Session.create(SessionId('bare'))
    expect(() => selectFoldRegion(bare, 0)).toThrow(RewindError)
    expect(() => selectFoldRegion(bare, 0)).toThrow(/no surface nodes after the checkpoint/)

    const session = explorationSession()
    // A mark appended last: nothing after it.
    session.append('checkpoint/mark', { turn: 1 })
    const mark = findLatestMark(session.events)!
    expect(() => selectFoldRegion(session, mark.seq)).toThrow(/no surface nodes after the checkpoint/)
  })

  it('rejects a fold with nothing between the mark and the rewind call', () => {
    // Build a fresh session where the mark lands right before the rewind call.
    const tight = Session.create(SessionId('tight'))
    tight.append('turn/start', { turn: 1 })
    tight.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'Find the cause' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    tight.append('checkpoint/mark', { turn: 1 })
    tight.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    const mark = findLatestMark(tight.events)!
    expect(() => selectFoldRegion(tight, mark.seq)).toThrow(/no surface nodes between the checkpoint/)
  })

  it('rejects an unbalanced region whose open tool call never resolves', () => {
    const session = Session.create(SessionId('unbalanced'))
    session.append('turn/start', { turn: 1 })
    session.append('checkpoint/mark', { turn: 1 })
    session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'bash') }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    const mark = findLatestMark(session.events)!
    expect(() => selectFoldRegion(session, mark.seq)).toThrow(/balanced boundary/)
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
    session.append('checkpoint/mark', { turn: 1 })
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [{ type: 'text', text: '{"seq":2}' }], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'bash') }, { surfaceOp: 'append' })
    session.append('tool/result', { turn: 1, step: 2, message: createToolResultMessage({ callId: CallId('c2'), content: [{ type: 'text', text: 'exploration output line' }], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 3, message: toolCallMessage('c3', 'rewind') }, { surfaceOp: 'append' })
    const mark = findLatestMark(session.events)!
    const region = selectFoldRegion(session, mark.seq)
    expect(region.shadowedSeqs).toEqual([5, 6])
    // The checkpoint call's own pair stays outside the fold and model-visible.
    expect(region.start).toBe(5)
    expect(region.end).toBe(6)
  })

  it('skips an orphaned result and folds nothing when only the rewind call follows', () => {
    const session = Session.create(SessionId('splitstart'))
    session.append('turn/start', { turn: 1 })
    session.append('checkpoint/mark', { turn: 1 })
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
    session.append('checkpoint/mark', { turn: 1 })
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c2'), content: [], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c3', 'rewind') }, { surfaceOp: 'append' })
    const mark = findLatestMark(session.events)!
    expect(() => selectFoldRegion(session, mark.seq)).toThrow(/no surface nodes between the checkpoint/)
  })

  it('rejects when an orphaned result is the only node after the mark', () => {
    const session = Session.create(SessionId('lone-result'))
    session.append('turn/start', { turn: 1 })
    session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'checkpoint') }, { surfaceOp: 'append' })
    session.append('checkpoint/mark', { turn: 1 })
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [], isError: false }) }, { surfaceOp: 'append' })
    const mark = findLatestMark(session.events)!
    expect(() => selectFoldRegion(session, mark.seq)).toThrow(/no surface nodes between the checkpoint/)
  })

  it('folds to the surface tail when no assistant message exists yet', () => {
    const session = Session.create(SessionId('noassistant'))
    session.append('checkpoint/mark', { turn: null })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'a' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    const region = selectFoldRegion(session, 0)
    expect(region.shadowedSeqs).toEqual([1])
  })

  it('asserts the fold lock bracket and the region stability', () => {
    const session = explorationSession()
    expect(() =>{  assertNoActiveFold(session.events) }).not.toThrow()
    session.append('checkpoint/fold-start', { turn: 1 })
    expect(() =>{  assertNoActiveFold(session.events) }).toThrow(/fold is already in progress/)
    session.append('checkpoint/fold-end', { turn: 1 })
    expect(() =>{  assertNoActiveFold(session.events) }).not.toThrow()

    const mark = findLatestMark(session.events)!
    const region = selectFoldRegion(session, mark.seq)
    expect(() =>{  assertRegionUnchanged(session, mark.seq, region) }).not.toThrow()
    expect(() =>{  assertRegionUnchanged(session, mark.seq, { ...region, end: region.end + 1 }) })
      .toThrow(/changed during summarization/)
  })

  it('tracks the open turn across boundaries', () => {
    const session = Session.create(SessionId('turns'))
    expect(openTurnOf(session.events)).toBeNull()
    session.append('turn/start', { turn: 7 })
    expect(openTurnOf(session.events)).toBe(7)
    session.append('turn/end', { turn: 7, reason: { kind: 'completed' } })
    expect(openTurnOf(session.events)).toBeNull()
  })
})

describe('rewind service', () => {
  it('resolves service config with defaults', () => {
    const read = (service: RewindService): { maxTokens: number; language: string } => ({
      maxTokens: (service as unknown as { maxTokens: number }).maxTokens,
      language: (service as unknown as { reportLanguage: string }).reportLanguage,
    })
    expect(read(new RewindService(new Context(), {}))).toEqual({ maxTokens: 1024, language: 'en' })
    expect(read(new RewindService(new Context(), { maxTokens: 512, reportLanguage: 'zh' })))
      .toEqual({ maxTokens: 512, language: 'zh' })
  })

  it('folds the exploration into an auto-generated report', async () => {
    const { ctx } = await setup(textChunks('## Findings\n- root cause is X'))
    const session = explorationSession()
    const result = await ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL)
    expect(result).toEqual({ checkpointSeq: 4, foldedNodes: 2, start: 5, end: 6 })

    const types = session.events.map(e => e.type)
    expect(types).toContain('checkpoint/fold-start')
    expect(types).toContain('checkpoint/fold-end')
    const record = session.events.find(e => e.type === 'checkpoint/rewind')
    expect(record?.data.report).toBe('## Findings\n- root cause is X')
    expect(record?.data.checkpointSeq).toBe(4)
    expect(record?.data.shadowedSeqs).toEqual([5, 6])
    expect(record?.data.shadowedRange).toEqual({ start: 5, end: 6 })

    const report = session.events.find(e => e.type === 'user/message'
      && e.data.source.kind === 'plugin' && e.data.source.plugin === 'rewind') as SessionEvent<'user/message'> | undefined
    expect(report?.surfaceOp).toEqual({ op: 'replace', start: 5, end: 6 })
    const messages = session.deriveMessages()
    expect(messages.map(m => m.content)).toEqual([
      [{ type: 'text', text: 'Find the cause' }],
      [{ type: 'text', text: '## Findings\n- root cause is X' }],
      [{ type: 'tool-call', id: CallId('c2'), name: 'rewind', arguments: '{}' }],
    ])
  })

  it('records provider usage in the rewind provenance', async () => {
    const chunks: StreamChunk[] = [
      ...textChunks('report').slice(0, 3),
      { type: 'usage', usage: { inputTokens: 1, outputTokens: 2 } },
      textChunks('report')[3]!,
    ]
    const { ctx } = await setup(chunks)
    const session = explorationSession()
    await ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL)
    const record = session.events.find(e => e.type === 'checkpoint/rewind')
    expect(record?.data.usage).toEqual({ inputTokens: 1, outputTokens: 2 })
  })

  it('routes the summarization request to the routed or configured model', async () => {
    const { ctx, adapter } = await setup(textChunks('report'))
    const session = explorationSession()
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
    const session = explorationSession()
    await ctx.rewind.rewind(agentWithSession(session), SIGNAL)
    expect(adapter.requests[0]!.provider).toBe('other')
    expect(adapter.requests[0]!.maxTokens).toBe(256)
    const last = adapter.requests[0]!.messages.at(-1)!
    expect(JSON.stringify(last.content)).toContain('探索报告引擎')
  })

  it('rejects a fold without a checkpoint mark', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = Session.create(SessionId('bare'))
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL)).rejects.toThrow(/no checkpoint\/mark/)
  })

  it('rejects a fold while another fold is in progress', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = explorationSession()
    session.append('checkpoint/fold-start', { turn: 1 })
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL)).rejects.toThrow(/fold is already in progress/)
  })

  it('rejects a summary that is not smaller than the folded region', async () => {
    const { ctx } = await setup(textChunks('x'.repeat(200)))
    const session = Session.create(SessionId('shrink'))
    session.append('turn/start', { turn: 1 })
    session.append('checkpoint/mark', { turn: 1 })
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
    session.append('checkpoint/mark', { turn: 1 })
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
    session.append('checkpoint/mark', { turn: 1 })
    session.append('assistant/message', { turn: 1, step: 1, message: toolCallMessage('c1', 'bash') }, { surfaceOp: 'append' })
    session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: CallId('c1'), content: [], isError: false }) }, { surfaceOp: 'append' })
    session.append('assistant/message', { turn: 1, step: 2, message: toolCallMessage('c2', 'rewind') }, { surfaceOp: 'append' })
    await expect(ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL))
      .resolves.toMatchObject({ foldedNodes: 2 })
  })

  it('records a fold-end with the error when summarization fails', async () => {
    const { ctx } = await setup([{ type: 'finish', reason: { kind: 'stop' } }])
    const session = explorationSession()
    await expect(ctx.rewind.rewind(agentWithSession(session), SIGNAL)).rejects.toThrow(/no text report/)
    const ends = session.events.filter(e => e.type === 'checkpoint/fold-end')
    expect(ends).toHaveLength(1)
    expect(ends[0]!.data.error).toMatch(/no text report/)
  })

  it('records a fold-end when the summarization stream errors', async () => {
    const { ctx } = await setup([{ type: 'finish', reason: { kind: 'error', failure: { message: 'boom', code: 'E' } } }])
    const session = explorationSession()
    await expect(ctx.rewind.rewind(agentWithSession(session, { provider: MODEL, model: MODEL }), SIGNAL))
      .rejects.toThrow(/stream ended with error/)
    const ends = session.events.filter(e => e.type === 'checkpoint/fold-end')
    expect(ends).toHaveLength(1)
    expect(ends[0]!.data.error).toMatch(/stream ended with error/)
  })

  it('propagates a cancelled signal and closes the bracket', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = explorationSession()
    const aborted = new AbortController()
    aborted.abort()
    await expect(ctx.rewind.rewind(agentWithSession(session), aborted.signal)).rejects.toBeDefined()
    const ends = session.events.filter(e => e.type === 'checkpoint/fold-end')
    expect(ends).toHaveLength(1)
  })

  it('rejects when no provider/model can be resolved for summarization', async () => {
    const { ctx } = await setup(textChunks('report'))
    const session = Session.create(SessionId('nohost'))
    session.append('checkpoint/mark', { turn: null })
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
      { maxTokens: 64, reportLanguage: 'en' },
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
      { maxTokens: 64, reportLanguage: 'en' },
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
      { maxTokens: 64, reportLanguage: 'en' },
    )).rejects.toThrow(message)
  })
})

describe('rewind tool', () => {
  async function setupTools(chunks: StreamChunk[]): Promise<Context> {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRegistry)
    await ctx.plugin(LlmService)
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
    const session = explorationSession()
    const result = await callRewind(ctx, agentWithSession(session, { provider: MODEL, model: MODEL }))
    const text = result.content.filter(b => b.type === 'text').map(b => b.text).join('')
    expect(text).toMatch(/Folded 2 surface nodes/)
    expect(session.events.some(e => e.type === 'checkpoint/rewind')).toBe(true)
  })

  it('presents the call and the result with a stable generic card', async () => {
    const ctx = await setupTools(textChunks('report'))
    const def = ctx.tools.get('rewind')!
    expect(def.presentCall?.({})).toEqual({ card: 'generic', title: 'Rewind', kind: 'other' })
    const rendered = def.output.render({}, { checkpointSeq: 4, foldedNodes: 2, start: 5, end: 6 })
    expect(rendered).toEqual([{ type: 'text', text: 'Folded 2 surface nodes (seq 5..6) into an auto-generated report.' }])
    const single = def.output.render({}, { checkpointSeq: 1, foldedNodes: 1, start: 2, end: 2 })
    expect(single).toEqual([{ type: 'text', text: 'Folded 1 surface node (seq 2..2) into an auto-generated report.' }])
  })

  it('registers the fold-discipline prompt section', async () => {
    const ctx = await setupTools(textChunks('report'))
    const assembly = await ctx.systemPrompt.assemble()
    const section = assembly.sections.find(s => s.name === 'tool:rewind')
    expect(section?.text).toContain('MUST call `rewind` immediately')
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
