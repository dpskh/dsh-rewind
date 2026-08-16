/**
 * Fold-region selection: the pure log/surface math a rewind runs. Selection
 * and validation are read-only; the mutation transaction lives in the
 * service. Boundary balance reuses the compaction seam's exported
 * `toolPairingBalancedBefore/After` so a fold never splits a tool-call/result
 * pair. Marks arrive as an anchor log length (from `ctx.checkpoint`), never
 * as session events.
 *
 * @module @dpskh/tool-rewind/region
 */

import { toolPairingBalancedAfter, toolPairingBalancedBefore } from '@deepseek-ai/dsh-compaction'
import type { Session } from '@deepseek-ai/dsh-session'

/** Stable failure classes for a rewind fold. */
export type RewindErrorCode =
  | 'NO_CHECKPOINT'
  | 'EMPTY_REGION'
  | 'UNBALANCED'
  | 'FOLD_IN_PROGRESS'
  | 'CHANGED'
  | 'SHRINK'

/** Typed error for rewind fold rejections. */
export class RewindError extends Error {
  override readonly name = 'RewindError'

  /**
   * Create one classified fold failure.
   * @param code - stable failure class.
   * @param message - backend diagnostic retained as the Error message.
   */
  constructor(
    readonly code: RewindErrorCode,
    message: string,
  ) {
    super(message)
  }
}

/** One validated inclusive span of current surface positions. */
export interface FoldRegion {
  /** Inclusive first surface-node seq. */
  readonly start: number
  /** Inclusive last surface-node seq. */
  readonly end: number
  /** The shadowed surface nodes, in surface order. */
  readonly shadowedSeqs: readonly number[]
}

/**
 * Select the fold region: every surface node at or after the mark anchor up
 * to the last balanced node before the current step's assistant message (the
 * message carrying the rewind call itself, which is never folded). The anchor
 * is the session log length at mark time — the seq of the first event after
 * the marker. A mark appended mid-call — the checkpoint tool runs between its
 * own assistant tool-call and tool/result — leaves an orphaned result as the
 * first node after the anchor; selection skips exactly that node, so the
 * checkpoint call's own pair stays visible and the fold starts at the
 * exploration that follows. Parallel tool calls issued alongside the
 * checkpoint (same assistant message) put further nodes after the anchor;
 * those belong to the exploration and are folded even though their leading
 * cut is unbalanced, so a fold never collapses to an empty region just
 * because the exploration ran in parallel with the checkpoint call. The
 * region must be non-empty and its trailing boundary must not split a
 * tool-call/result pair.
 * @param session - the session whose surface to fold.
 * @param markLogLength - the session log length at mark time (fold anchor).
 * @returns the validated inclusive surface span.
 * @throws {@link RewindError} `EMPTY_REGION` / `UNBALANCED` when no valid
 * span exists.
 */
export function selectFoldRegion(session: Session, markLogLength: number): FoldRegion {
  const nodes = session.surface.nodes
  let startIdx = nodes.findIndex(seq => seq >= markLogLength)
  if (startIdx === -1) {
    throw new RewindError('EMPTY_REGION', 'rewind: no surface nodes after the checkpoint mark')
  }
  // The mark lands between the checkpoint call's own tool-call and result, so
  // the first node after it is that orphaned result with an unbalanced leading
  // cut. Skip exactly that node. Do NOT skip ahead to the next balanced cut:
  // parallel exploration calls issued in the same assistant message as the
  // checkpoint leave their results behind the mark too, and skipping to a
  // balanced cut would jump past the whole exploration to the rewind call.
  /* oxlint-disable-next-line typescript/no-non-null-assertion -- startIdx is clamped to nodes.length, so the node exists */
  if (startIdx < nodes.length && !toolPairingBalancedBefore(session, nodes[startIdx]!)) {
    startIdx += 1
  }
  const events = session.events
  const currentMessageIdx = events.findLastIndex(event => event.type === 'assistant/message')
  // seq === array index (the append contract), so seqs compare to indexes.
  let endIdx = nodes.length - 1
  if (currentMessageIdx !== -1) {
    /* oxlint-disable-next-line typescript/no-non-null-assertion -- endIdx is clamped to startIdx, so the node exists */
    while (endIdx >= startIdx && nodes[endIdx]! >= currentMessageIdx) endIdx -= 1
  }
  if (endIdx < startIdx) {
    throw new RewindError(
      'EMPTY_REGION',
      'rewind: no surface nodes between the checkpoint mark and the rewind call',
    )
  }
  /* oxlint-disable-next-line typescript/no-non-null-assertion -- endIdx is clamped to startIdx, so the node exists */
  while (endIdx >= startIdx && !toolPairingBalancedAfter(session, nodes[endIdx]!)) endIdx -= 1
  if (endIdx < startIdx) {
    throw new RewindError(
      'UNBALANCED',
      'rewind: the fold region cannot be cut at a balanced boundary (an open tool-call/result pair spans it)',
    )
  }
  return {
    // oxlint-disable-next-line typescript/no-non-null-assertion -- both indexes are valid surface positions by construction
    start: nodes[startIdx]!,
    /* oxlint-disable-next-line typescript/no-non-null-assertion -- endIdx is a valid surface position by construction */
    end: nodes[endIdx]!,
    shadowedSeqs: nodes.slice(startIdx, endIdx + 1),
  }
}

/**
 * Reject a fold whose region moved after it was selected — the asynchronous
 * summarization must not commit a replacement over a span that changed
 * underneath it (a concurrent compaction fold, for example).
 * @param session - the session being folded.
 * @param markLogLength - the session log length at mark time (fold anchor).
 * @param region - the region selected before the summarization.
 * @throws {@link RewindError} `CHANGED` when the reselected span differs.
 */
export function assertRegionUnchanged(
  session: Session,
  markLogLength: number,
  region: FoldRegion,
): void {
  const reselected = selectFoldRegion(session, markLogLength)
  if (reselected.start !== region.start
    || reselected.end !== region.end
    || reselected.shadowedSeqs.length !== region.shadowedSeqs.length) {
    throw new RewindError('CHANGED', 'rewind: the fold region changed during summarization')
  }
}
