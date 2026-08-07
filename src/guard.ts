/**
 * The turn-ending fold guard: an active (unfolded) checkpoint must not end
 * the turn. When the model tries to yield with the most recent mark unfolded,
 * `agent/turn-stopping` injects a standing warning into the agent's next-step
 * inbox — the turn loop keeps running while that inbox is non-empty — so the
 * model must call `rewind` before the turn closes. One warning per agent per
 * turn, so a rewind that legitimately fails (e.g. an empty region) cannot
 * trap the turn in a warning loop.
 *
 * @module @dpskh/tool-rewind/guard
 */

import type { Context } from 'cordis'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { hasActiveCheckpoint } from './region.ts'

/** Canonical source for the injected guard warning (same marker family as the fold report). */
export const REWIND_GUARD_SOURCE = Object.freeze({ kind: 'plugin', plugin: 'rewind' } as const)

/** The warning the model sees when it tries to end a turn with an unfolded checkpoint. */
export const REWIND_GUARD_TEXT = [
  'You are in an active checkpoint: the exploration after the `checkpoint` mark has not been folded.',
  'You MUST call `rewind` before ending this turn — it folds the exploration into an auto-generated report.',
  'Do NOT end the turn without completing the checkpoint.',
].join(' ')

/**
 * Install the fold guard: intercept `agent/turn-stopping` and keep the turn
 * alive while the latest checkpoint mark is unfolded.
 * @param ctx - context whose agent turns are guarded.
 */
export function installRewindGuard(ctx: Context): void {
  const warned = new Set<string>()
  ctx.on('agent/turn-stopping', ({ agent, turn }) => {
    const key = `${agent.id}:${turn}`
    if (warned.has(key)) return
    if (!hasActiveCheckpoint(agent.session.events)) return
    warned.add(key)
    agent.inject(createUserMessage({
      content: [{ type: 'text', text: REWIND_GUARD_TEXT }],
      source: REWIND_GUARD_SOURCE,
    }))
  })
}
