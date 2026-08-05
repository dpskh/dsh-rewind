/**
 * The report message's canonical source marker plus its predicate: the
 * replacement `user/message` a rewind lands carries this source so consumers
 * (UI, replay, title generation) recognize it as a fold report rather than a
 * verbatim user prompt. This module is cordis-free (no module augmentation)
 * so client and wire programs can name the marker without loading the host
 * plugin's Context merges — the `dsh-compact/checkpoint` shape.
 *
 * @module @dpskh/tool-rewind/brand
 */

import type { MessageSource } from '@deepseek-ai/dsh-llm/message'

/** Canonical source for the replacement user message produced by every rewind fold. */
export const REWIND_REPORT_SOURCE = Object.freeze({ kind: 'plugin', plugin: 'rewind' } as const)

/**
 * Test whether a persisted message source identifies a rewind report.
 * @param source - source restored from a surface user message.
 * @returns whether the source carries the fold report marker.
 */
export function isRewindReportSource(source: MessageSource): boolean {
  return source.kind === 'plugin' && source.plugin === REWIND_REPORT_SOURCE.plugin
}
