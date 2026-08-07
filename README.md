# @dpskh/tool-rewind — exploration fold for the DeepSeek Harness

English | [中文](README.zh.md)

One package, one entry plugin. Mounting `@dpskh/tool-rewind` provides `ctx.rewind` (a service that folds everything since the most recent `checkpoint/mark` into an auto-generated report) and the model-facing `rewind` tool over it. The fold shadows the exploration's surface range with the report — the same `surfaceOp: { op: 'replace' }` mechanism the DeepSeek Harness compaction seam uses — so subsequent requests no longer carry the exploration's intermediate steps (reads, searches, experiments), while the durable log keeps the full exploration for audit. The report is produced by `ctx.llm` (auto-generated); the marker comes from the sibling [`@dpskh/tool-checkpoint`](https://github.com/dsh-external/dsh-checkpoint) plugin. No upstream package is modified.

## Configuration

```yaml
- id: tool-rewind
  name: '@dpskh/tool-rewind'
  config:
    toolName: rewind                # model-facing tool name (default rewind)
    maxTokens: 1024                 # summarization output budget
    summarizationProvider: deepseek # optional override of the routed provider
    summarizationModel: deepseek-chat
    reportLanguage: en              # en | zh (report instruction language)
```

Provider/model resolution order for the report call: configured overrides, then the session's last routed `request/header` config, then the agent's options. Missing all three fails loud.

## Working together

The fold needs a marker: mount both plugins so `rewind` can collapse the exploration anchored by a `checkpoint`. `rewind` alone has nothing to fold and fails with a no-checkpoint error; `checkpoint` alone records inert markers.

```yaml
- id: tool-checkpoint
  name: '@dpskh/tool-checkpoint'    # https://github.com/dsh-external/dsh-checkpoint
- id: tool-rewind
  name: '@dpskh/tool-rewind'
  config:
    reportLanguage: en              # en | zh report instruction language
```

## Contract

- `ctx.rewind.rewind(agent, signal)` — find the most recent `checkpoint/mark`; select the balanced surface span after it (never splitting a tool-call/result pair, never folding the rewind call itself); summarize the span over `ctx.llm`; then commit, in one synchronous block, a `checkpoint/rewind` provenance record plus a replacement `user/message` carrying the report with source marker `{ kind: 'plugin', plugin: 'rewind' }` (`isRewindReportSource` recognizes it). Failures close the fold bracket with an error record and reject with a classified `RewindError` (`NO_CHECKPOINT`, `EMPTY_REGION`, `UNBALANCED`, `FOLD_IN_PROGRESS`, `CHANGED`, `SHRINK`). A mark appended mid-call — the checkpoint tool runs between its own tool-call and result — leaves that result orphaned right after the mark; selection skips it to the next balanced cut, so the checkpoint call's own pair stays visible and the fold starts at the exploration that follows.
- `rewind` tool — no arguments → `{ checkpointSeq, foldedNodes, start, end }`. Render intent: generic card.
- **Turn guard** — while the latest `checkpoint/mark` is unfolded, `agent/turn-stopping` injects a standing warning into the agent's next-step inbox (one per agent per turn), so the turn cannot end until `rewind` folds the exploration. A rewind that legitimately fails (e.g. an empty region) cannot trap the turn in a warning loop.

## Model Experience

Directly: the `rewind` tool call and its fold result. The replacement report message is model-visible from the next request on; the exploration's durable events stay in the log but out of the model-visible surface. A `tool:rewind` system-prompt section makes fold discipline a standing instruction (rewind immediately after the marked exploration, before unrelated work).

#### KV Cache effect

The report call reuses the session's system prompt and tool schemas so it stays a prefix of the last routed request (warm prefix cache). The fold itself shifts the model-visible prefix of subsequent requests by design.

## Known Limitations and Deferred Work

- **Shrink check is char-based** — the report must be shorter than the folded region's text (measured in characters, not tokens); a token-meter-based check would be more precise.
- **One fold per marker** — folding the same marker twice is rejected as an empty region; a new exploration needs a fresh `checkpoint`.
- **Mid-exploration compaction** — if automatic compaction shadows part of the exploration before `rewind` runs, the fold covers whatever remains on the surface; the log-only marker survives either way.
