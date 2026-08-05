/**
 * Rewind vocabulary: the `checkpoint/*` session events this plugin appends.
 * `checkpoint/mark` is a mirror of the identical declaration in
 * `@dpskh/tool-checkpoint/src/types.ts` (the canonical shape home) — the two
 * plugins stay independent, and TypeScript merges the identical members
 * without conflict when both are loaded.
 *
 * @module @dpskh/tool-rewind/types
 */

import type { TokenUsage } from '@deepseek-ai/dsh-llm'

declare module '@deepseek-ai/dsh-session' {
  interface SessionEventMap {
    /**
     * Log-only marker (canonical declaration:
     * `@dpskh/tool-checkpoint/src/types.ts`): records the conversation
     * position an exploration starts from. Deliberately carries no
     * `surfaceOp` so compaction or a fold never shadows it.
     */
    'checkpoint/mark': {
      turn: number | null
      objective?: string
    }
    /**
     * Log-only bracket opener: holds the fold lock until the matching
     * `checkpoint/fold-end`. A numbered `turn` is the open turn the fold runs
     * in; `null` identifies an idle-session fold. An unmatched start means a
     * fold attempt failed before it could close (failed attempts remain
     * visible in the log).
     */
    'checkpoint/fold-start': {
      turn: number | null
    }
    /**
     * Provenance record of a completed fold — log-only, no `surfaceOp`. The
     * report text is in `data.report`; the actual surface replacement is a
     * subsequent `user/message` event that shadows the folded range and
     * carries this record's seq in its `sourceEventSeqs`.
     */
    'checkpoint/rewind': {
      turn: number | null
      report: string
      checkpointSeq: number
      shadowedRange: { start: number; end: number }
      shadowedSeqs: number[]
      provider: string
      model: string
      usage?: TokenUsage
    }
    /**
     * Log-only bracket closer; `error` records a failed fold attempt.
     */
    'checkpoint/fold-end': {
      turn: number | null
      error?: string
    }
  }
}

/** Re-export the merged map member types for consumers. */
export type CheckpointRewindEvent = import('@deepseek-ai/dsh-session').SessionEvent<'checkpoint/rewind'>
