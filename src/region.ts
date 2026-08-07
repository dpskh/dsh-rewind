/**
 * Fold-region selection: the pure log/surface math a rewind runs. Selection
 * and validation are read-only; the mutation transaction lives in the
 * service. Boundary balance reuses the compaction seam's exported
 * `toolPairingBalancedBefore/After` so a fold never splits a tool-call/result
 * pair.
 *
 * @module @dpskh/tool-rewind/region
 */

import { toolPairingBalancedAfter, toolPairingBalancedBefore } from '@deepseek-ai/dsh-compact'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'

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

/** The open turn enclosing a log position, or `null` between turns. */
export function openTurnOf(events: readonly SessionEvent[]): number | null {
  let latest: SessionEvent<'turn/start'> | SessionEvent<'turn/end'> | undefined
  for (const event of events) {
    if (event.type === 'turn/start' || event.type === 'turn/end') latest = event
  }
  return latest?.type === 'turn/start' ? latest.data.turn : null
}

/**
 * The most recent checkpoint mark, or `undefined` when none exists.
 * @param events - a session log.
 * @returns the latest `checkpoint/mark` event, or `undefined`.
 */
export function findLatestMark(
  events: readonly SessionEvent[],
): SessionEvent<'checkpoint/mark'> | undefined {
  return events.findLast(event => event.type === 'checkpoint/mark')
}

/**
 * Reject a fold while another is in progress: any unmatched
 * `checkpoint/fold-start` bracket in the log.
 * @param events - a session log.
 * @throws {@link RewindError} `FOLD_IN_PROGRESS` when a bracket is open.
 */
export function assertNoActiveFold(events: readonly SessionEvent[]): void {
  let open = 0
  for (const event of events) {
    if (event.type === 'checkpoint/fold-start') open += 1
    if (event.type === 'checkpoint/fold-end') open = Math.max(0, open - 1)
  }
  if (open > 0) {
    throw new RewindError(
      'FOLD_IN_PROGRESS',
      'rewind: another fold is already in progress; the session fold lock is already active',
    )
  }
}

/**
 * Select the fold region: every surface node after the checkpoint mark up to
 * the last balanced node before the current step's assistant message (the
 * message carrying the rewind call itself, which is never folded). A mark
 * appended mid-call — the checkpoint tool runs between its own assistant
 * tool-call and tool/result — leaves an orphaned result as the first node
 * after the mark; selection skips past such nodes to the next balanced cut,
 * so the fold never splits the checkpoint call's own pair. The region must
 * be non-empty and its boundaries must not split a tool-call/result pair.
 * @param session - the session whose surface to fold.
 * @param markSeq - seq of the `checkpoint/mark` the fold starts after.
 * @returns the validated inclusive surface span.
 * @throws {@link RewindError} `EMPTY_REGION` / `UNBALANCED` when no valid
 * span exists.
 */
export function selectFoldRegion(session: Session, markSeq: number): FoldRegion {
  const nodes = session.surface.nodes
  let startIdx = nodes.findIndex(seq => seq > markSeq)
  if (startIdx === -1) {
    throw new RewindError('EMPTY_REGION', 'rewind: no surface nodes after the checkpoint mark')
  }
  while (startIdx < nodes.length) {
    /* oxlint-disable-next-line typescript/no-non-null-assertion -- startIdx is clamped to nodes.length, so the node exists */
    if (toolPairingBalancedBefore(session, nodes[startIdx]!)) break
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
 * @param markSeq - seq of the `checkpoint/mark` the fold starts after.
 * @param region - the region selected before the summarization.
 * @throws {@link RewindError} `CHANGED` when the reselected span differs.
 */
export function assertRegionUnchanged(
  session: Session,
  markSeq: number,
  region: FoldRegion,
): void {
  const reselected = selectFoldRegion(session, markSeq)
  if (reselected.start !== region.start
    || reselected.end !== region.end
    || reselected.shadowedSeqs.length !== region.shadowedSeqs.length) {
    throw new RewindError('CHANGED', 'rewind: the fold region changed during summarization')
  }
}
