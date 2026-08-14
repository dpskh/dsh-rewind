/**
 * Package-owned invariant companion for `@dpskh/tool-rewind`.
 * @module @dpskh/tool-rewind/invariant
 */

/* jscpd:ignore-start */
import type { Context } from '@deepseek-ai/cordis'
import type { InvariantFailure, InvariantInstaller } from '@deepseek-ai/dsh-invariants'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'

const PACKAGE_NAME = '@dpskh/tool-rewind'

// Re-export pulls the declaration-merged event map into this companion's
// program, so the spec can name the `checkpoint/*` events.
export type { CheckpointRewindEvent } from './types.ts'

/** Cordis companion plugin name. */
export const name = 'tool-rewind-invariant'
/** Service required before the companion can reserve package ownership. */
export const inject = ['invariants']

/**
 * Assert the fold bracket stays balanced and every `checkpoint/rewind`
 * record references a real `checkpoint/mark`: fold-start and fold-end events
 * alternate without a negative depth, and the record's `checkpointSeq` names
 * a mark present in the owning session's log (checked against the live
 * session, so resumed sessions whose marks arrived as constructor seeds still
 * verify).
 */
const install: InvariantInstaller = (ctx: Context, fail: InvariantFailure) => {
  let openFolds = 0
  ctx.on('session/event', (session: Session, event: SessionEvent) => {
    if (event.type === 'checkpoint/fold-start') openFolds += 1
    if (event.type === 'checkpoint/fold-end') {
      if (openFolds === 0) {
        fail(`checkpoint/fold-end at seq ${event.seq} closes a fold that is not open`)
      }
      openFolds -= 1
    }
    if (event.type === 'checkpoint/rewind') {
      const mark = session.events[event.data.checkpointSeq]
      if (mark?.type !== 'checkpoint/mark') {
        fail(`checkpoint/rewind at seq ${event.seq} references seq ${event.data.checkpointSeq} which is not a checkpoint/mark`)
      }
    }
  }, { global: true })
}

/**
 * Register this package's invariant companion.
 * @param ctx - Cordis context carrying the invariant service.
 * @returns the installed registration's disposer after setup succeeds.
 */
export const apply = (ctx: Context): Promise<() => void> =>
  Promise.resolve(ctx.invariants.register(PACKAGE_NAME, install))
/* jscpd:ignore-end */
