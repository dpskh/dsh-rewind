/**
 * Auto-generated fold reports: the summarization input built from a fold
 * region and the `ctx.llm.stream` call that produces the report. The input
 * reproduces the session's own system prompt and tool schemas so the
 * auxiliary call stays a prefix of the last routed request and reuses the
 * provider's warm prefix cache — the compaction summarizer's trick.
 *
 * @module @dpskh/tool-rewind/summarize
 */

import type { Context } from '@deepseek-ai/cordis'
import { BlockAssembler, createUserMessage, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  Message,
  TokenUsage,
  ToolSchema,
} from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session } from '@deepseek-ai/dsh-session'
import type { FoldRegion } from './region.ts'

/** The replayed conversation surface the summarizer condenses. */
export interface RewindSummarizationInput {
  /** The conversation's own system prompt, reused for prefix-cache alignment. */
  readonly system?: string
  /** The conversation's tool schemas, reused for prefix-cache alignment. */
  readonly tools?: readonly ToolSchema[]
  /** The fold region, in surface order, that precedes the instruction. */
  readonly messages: readonly Message[]
}

/** Resolved rewind summarization settings. */
export interface RewindSummarizeConfig {
  readonly maxTokens: number
  readonly maxSummarizationRetries: number
  readonly summarizationProvider?: string
  readonly summarizationModel?: string
  readonly reportLanguage: 'en' | 'zh'
}

/** Safe report content plus the exact auxiliary call envelope. */
export interface RewindSummary {
  /** Text-only report that replaces the folded region. */
  report: string
  /** Provider the report was routed to. */
  provider: string
  /** Model the report was routed to. */
  model: string
  /** Max tokens the auxiliary call was bounded by. */
  maxTokens: number
  /** Provider token accounting when the adapter reported any. */
  usage?: TokenUsage
}

/** The summarization directive, delivered as the final user message. */
const REWIND_INSTRUCTION_EN = [
  'You are acting as an exploration-report engine for this AI coding assistant. Condense the exploration transcript ABOVE into a concise report that lets the assistant continue the task without the exploration\'s intermediate steps.',
  '',
  'Output EXACTLY the Markdown structure below; keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Findings',
  '- [what the exploration established: conclusions, evidence, numbers]',
  '',
  '## Files and Artifacts',
  '- [exact path: why it matters, key changes or snippets]',
  '',
  '## Decisions',
  '- [choices made during the exploration and their rationale]',
  '',
  '## Unresolved Questions',
  '- [open questions the exploration could not settle]',
  '',
  '## Recommended Next Step',
  '- [the single next action, or "(none)"]',
  '',
  'Rules:',
  '- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.',
  '- Report the exploration\'s conclusions and evidence, not its process.',
  '- Do NOT mention this summarization request or that the context was compacted.',
  '- Output only the report text: do not call any tool or take any other action.',
].join('\n')

/** The Chinese-language summarization directive. */
const REWIND_INSTRUCTION_ZH = [
  '你正在为这个 AI 编码助手充当探索报告引擎。请把上面这段探索记录压缩成一份简洁报告，使助手无需探索的中间步骤即可继续任务。',
  '',
  '严格按下面的 Markdown 结构输出，各节顺序不变；用简洁的要点而非整段叙述；空节写 "(none)"，不要删节。',
  '',
  '## 发现',
  '- [探索确定了什么：结论、证据、数字]',
  '',
  '## 文件与产物',
  '- [精确路径：为何重要、关键改动或片段]',
  '',
  '## 决策',
  '- [探索期间做出的选择及其理由]',
  '',
  '## 未决问题',
  '- [探索未能解决的开放问题]',
  '',
  '## 建议的下一步',
  '- [唯一的下一个动作，或 "(none)"]',
  '',
  '规则：',
  '- 使用简洁的中文工程表述。保留精确的文件路径、命令、错误字符串、标识符、数值、函数签名与语法片段。',
  '- 报告探索的结论与证据，而非其过程。',
  '- 不要提及本次总结请求或上下文已被压缩。',
  '- 只输出报告文本：不要调用任何工具或采取其他动作。',
].join('\n')

/**
 * Build the summarizer's input from a fold region: the region's derived
 * messages plus the session's current system prompt and tool schemas.
 * @param session - the session being folded.
 * @param region - the validated fold span.
 * @returns the replay input the auxiliary call consumes.
 */
export function buildSummarizationInput(
  session: Session,
  region: FoldRegion,
): RewindSummarizationInput {
  const header = session.requestHeader()
  const messages: Message[] = []
  for (const seq of region.shadowedSeqs) {
    /* oxlint-disable-next-line typescript/no-non-null-assertion -- shadowedSeqs are surface positions, valid log indexes */
    const message = session.deriveEventMessage(session.events[seq]!)
    if (message !== null) messages.push(message)
  }
  return {
    ...header?.system === undefined ? {} : { system: header.system },
    ...header?.tools === undefined ? {} : { tools: header.tools },
    messages,
  }
}

/** Join the text blocks of a message list (the shrink-check yardstick). */
export function regionTextLength(messages: readonly Message[]): number {
  let length = 0
  for (const message of messages) {
    for (const block of message.content) {
      if (block.type === 'text') length += block.text.length
    }
  }
  return length
}

/**
 * Run the one-shot report summarization over `ctx.llm`. Provider/model
 * resolution order: configured overrides, the session's last routed
 * `request/header` config, then the agent's options.
 * @param ctx - context carrying the llm service.
 * @param input - the replay input preceding the instruction.
 * @param agent - the agent whose session is folded (routing fallback).
 * @param config - resolved summarization settings.
 * @param signal - optional cancellation signal forwarded to the stream.
 * @returns the text-only report and the exact auxiliary call envelope.
 */
export async function summarizeRegion(
  ctx: Context,
  input: RewindSummarizationInput,
  agent: Agent,
  config: RewindSummarizeConfig,
  signal?: AbortSignal,
): Promise<RewindSummary> {
  const latest = agent.session.requestHeader()?.config
  const configured = config.summarizationProvider === undefined || config.summarizationModel === undefined
    ? undefined
    : { provider: config.summarizationProvider, model: config.summarizationModel }
  const agentTarget = agent.options.provider !== undefined
    && agent.options.provider.length > 0
    && agent.options.model !== undefined
    && agent.options.model.length > 0
    ? { provider: agent.options.provider, model: agent.options.model }
    : undefined
  const target = configured ?? latest ?? agentTarget
  if (target === undefined) {
    throw new Error(
      'no provider/model available for rewind summarization: set both RewindConfig summarization fields, route one request, or set both AgentOptions fields',
    )
  }

  // Condensing an exploration is mechanical; a reasoning model's thinking
  // tokens otherwise compete with the report text under maxTokens and can
  // exhaust the budget before any text starts (finish `length` with an
  // empty report — the repeated "no text report content" failures). Disable
  // thinking when the route exposes an `off` reasoning effort; routes
  // without one keep their default effort.
  let reasoningEffort: ReasoningEffortId | undefined
  try {
    const info = await ctx.llm.resolveModelInfo(target.provider, target.model, signal)
    if (info.reasoning?.efforts.some(effort => effort.id === OFF_REASONING_EFFORT)) {
      reasoningEffort = OFF_REASONING_EFFORT
    }
  } catch {
    // Capability probing must never fail the fold: without reasoning
    // metadata the call keeps the route's default effort.
  }

  const instruction = config.reportLanguage === 'zh' ? REWIND_INSTRUCTION_ZH : REWIND_INSTRUCTION_EN
  const messages: Message[] = [
    ...input.messages,
    createUserMessage({
      content: [{ type: 'text', text: instruction }],
      source: { kind: 'plugin', plugin: 'rewind' },
    }),
  ]
  const options: GenerateOptions = {
    provider: target.provider,
    model: target.model,
    messages,
    ...input.system === undefined ? {} : { system: input.system },
    ...input.tools === undefined ? {} : { tools: [...input.tools] },
    maxTokens: config.maxTokens,
    sessionId: agent.session.id,
    ...reasoningEffort === undefined ? {} : { reasoningEffort },
    ...signal === undefined ? {} : { signal },
  }
  // An empty completion is sampling noise, not a bad region: retry it, but
  // never an error/aborted finish or a cancelled signal.
  for (let attempt = 0; ; attempt++) {
    const assembler = new BlockAssembler()
    for await (const chunk of ctx.llm.stream(options)) assembler.push(chunk)
    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') {
      throw new Error(`rewind summarization stream ended with ${finish.kind}`)
    }
    const report = assembler.blocks()
      .filter((block): block is Extract<typeof block, { type: 'text' }> => block.type === 'text')
      .map(block => block.text)
      .join('')
    if (report.trim().length > 0) {
      return {
        report,
        provider: options.provider,
        model: options.model,
        maxTokens: config.maxTokens,
        ...assembler.usage === undefined ? {} : { usage: assembler.usage },
      }
    }
    signal?.throwIfAborted()
    if (attempt >= config.maxSummarizationRetries) {
      const usage = assembler.usage
      const detail = usage === undefined
        ? `finish: ${finish.kind}`
        : `finish: ${finish.kind}, outputTokens: ${usage.outputTokens}`
          + (usage.reasoningTokens === undefined ? '' : `, reasoningTokens: ${usage.reasoningTokens}`)
      throw new Error(
        `rewind summarization produced no text report content after ${config.maxSummarizationRetries} retries (${detail})`,
      )
    }
  }
}

/** The reasoning effort that disables thinking, as advertised by reasoning routes. */
const OFF_REASONING_EFFORT = ReasoningEffortId('off')
