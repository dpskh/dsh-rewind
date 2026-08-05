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
export const inject = ['tools', 'rewind']

/**
 * Register the `rewind` tool: folds everything since the most recent
 * `checkpoint` mark into an auto-generated report. Render intent: generic.
 */
export function apply(ctx: Context, config: ToolConfig): void {
  ctx.tools.register(defineTool({
    name: config.toolName,
    description:
      'Fold everything since the most recent `checkpoint` mark into an auto-generated report. The report replaces the exploration in the context from now on — its noisy middle (reads, searches, experiments) stops costing tokens, while the durable log keeps the full exploration for audit. Call it once an exploration marked with `checkpoint` is done.',
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
