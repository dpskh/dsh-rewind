/**
 * The model-facing `rewind` tool as its own child plugin: mounted by the
 * entry plugin after `ctx.rewind` exists, so it can inject the service it
 * delegates to (a plugin cannot inject a service it provides itself).
 *
 * @module @dpskh/tool-rewind/tool
 */

import type { Context } from 'cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'

/** Tool-plugin config: the model-facing tool name (default `rewind`). */
export interface ToolConfig {
  toolName: string
}

/** Cordis plugin name used by loader diagnostics. */
export const name = 'tool-rewind-tool'
export const inject = ['tools', 'rewind', 'systemPrompt']

/**
 * Cross-call fold discipline: rewind as soon as the marked exploration is
 * done, before unrelated work starts. Standing guidance lives in the prompt
 * (not the one-call schema prose) so the workflow is an instruction the
 * model follows by default.
 */
const REWIND_PROMPT_TEXT =
  'Fold discipline: you MUST call `rewind` as soon as a marked exploration is done, before starting unrelated work — it folds the exploration\'s intermediate steps into an auto-generated report (conclusions, evidence, exact paths). Without a preceding `checkpoint` mark it errors, so never call it blindly. A failed fold is contained and retryable; the region stays intact.'

/**
 * Register the `rewind` tool: folds everything since the most recent
 * `checkpoint` mark into an auto-generated report. Render intent: generic.
 */
export function apply(ctx: Context, config: ToolConfig): void {
  ctx.systemPrompt.section({
    name: 'tool:rewind',
    order: 151,
    text: REWIND_PROMPT_TEXT,
  })

  ctx.tools.register(defineTool({
    name: config.toolName,
    description:
      'End an active checkpoint: fold everything since the most recent `checkpoint` mark into an auto-generated report, replacing the exploration\'s intermediate steps in the context. The current call and its results are never folded.\n'
      + '\n'
      + 'Call as soon as the marked investigation is done, before starting unrelated work.\n'
      + '\n'
      + 'Requirements:\n'
      + '- You MUST call this before finishing your turn if a `checkpoint` mark is active.\n'
      + '- Call it as soon as the exploration is done — do not start unrelated work first.\n'
      + '- The report is auto-generated; it keeps conclusions, evidence, and exact paths.\n'
      + '\n'
      + 'Behavior:\n'
      + '- If no `checkpoint` mark exists, this tool errors with a no-checkpoint error.\n'
      + '- A failed fold is contained: the region stays intact and the fold may be retried.\n'
      + '- A successful rewind is final for that mark; repeat calls error with an empty region — continue from the retained report instead of retrying.',
    parameters: {},
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          checkpointSeq: { type: 'integer', required: true },
          foldedNodes: { type: 'integer', required: true },
          start: { type: 'integer', required: true },
          end: { type: 'integer', required: true },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: `Folded ${value.foldedNodes} surface node${value.foldedNodes === 1 ? '' : 's'} (seq ${value.start}..${value.end}) into an auto-generated report.`,
      }],
    },
    execute: async (_args, exec) => {
      if (exec.agent === undefined) {
        // The fold is per-session state; a non-agent caller has no session.
        throw new Error('rewind requires an owning agent session')
      }
      return ctx.rewind.rewind(exec.agent, exec.signal)
    },
    presentCall: () => ({ card: 'generic', title: 'Rewind', kind: 'other' }),
  }))
}
