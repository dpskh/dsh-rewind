# @dpskh/tool-rewind —— DeepSeek Harness 的探索折叠插件

[English](README.md) | 中文

一个包、一个入口插件。挂载 `@dpskh/tool-rewind` 提供 `ctx.rewind`（把最近一次 `checkpoint/mark` 之后的一切折叠成自动生成报告的服务）和基于它的模型侧 `rewind` 工具。折叠用报告遮蔽探索的表面区间 —— 与 DeepSeek Harness 压缩接缝相同的 `surfaceOp: { op: 'replace' }` 机制 —— 使后续请求不再携带探索的中间步骤（读文件、搜索、试验），而持久日志完整保留探索过程供审计。报告由 `ctx.llm` 自动生成；标记来自兄弟插件 [`@dpskh/tool-checkpoint`](https://github.com/dsh-external/dsh-checkpoint)。不修改任何上游包。

## 配置

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

报告调用的 provider/model 解析顺序：配置覆盖，其次会话最近路由的 `request/header` 配置，最后 agent 的 options。三者全缺则响亮报错。

## 配合使用

折叠需要标记：两个插件一起挂载，`rewind` 才能折叠 `checkpoint` 锚定的探索。单独挂 `rewind` 没有可折叠的内容，会以 no-checkpoint 错误失败；单独挂 `checkpoint` 只会记录惰性标记。

```yaml
- id: tool-checkpoint
  name: '@dpskh/tool-checkpoint'    # https://github.com/dsh-external/dsh-checkpoint
- id: tool-rewind
  name: '@dpskh/tool-rewind'
  config:
    reportLanguage: en              # en | zh report instruction language
```

## 契约

- `ctx.rewind.rewind(agent, signal)` —— 找到最近的 `checkpoint/mark`；选取其后的平衡表面区间（绝不切开 tool-call/result 配对，绝不折叠 rewind 调用本身）；通过 `ctx.llm` 总结该区间；随后在**一个同步块**内提交 `checkpoint/rewind` 溯源记录和携带报告的替换 `user/message`（source 标记 `{ kind: 'plugin', plugin: 'rewind' }`，`isRewindReportSource` 可识别）。失败时以错误记录闭合折叠括号，并以分类 `RewindError`（`NO_CHECKPOINT`、`EMPTY_REGION`、`UNBALANCED`、`FOLD_IN_PROGRESS`、`CHANGED`、`SHRINK`）拒绝。若标记在调用中途追加（checkpoint 工具在自身的 tool-call 与结果之间运行），该结果会成为标记后的孤立节点；选区会跳过它到下一个平衡切点，使 checkpoint 调用自身的配对保持可见，折叠从随后的探索开始。
- `rewind` 工具 —— 无参数 → `{ checkpointSeq, foldedNodes, start, end }`。渲染意图：generic 卡片。
- **回合守卫** —— 当最近的 `checkpoint/mark` 尚未折叠时，`agent/turn-stopping` 会向 agent 的 next-step 收件箱注入常驻警告（每个 agent 每个回合一次），使回合无法在 `rewind` 折叠探索之前结束。合理失败的 rewind（如空区域）不会把回合困在警告循环里。

## 模型体验

直接可见：`rewind` 工具调用及其折叠结果。替换后的报告消息从下一次请求起对模型可见；探索的持久事件仍在日志中，但不在模型可见表面。`tool:rewind` 系统提示 section 将折叠纪律变为常驻指令（标记的探索一结束立即 rewind，先于无关工作）。

#### KV 缓存影响

报告调用复用会话的系统提示词与工具 schema，保持为最近一次路由请求的前缀（热前缀缓存）。折叠本身按设计改变后续请求的模型可见前缀。

## 已知限制与待办

- **收缩检查基于字符** —— 报告必须比折叠区域的文本更短（按字符而非 token 度量）；基于 token 计量的检查会更精确。
- **每个标记只能折叠一次** —— 对同一标记二次折叠会因区域为空而被拒绝；新探索需要新的 `checkpoint`。
- **探索中途发生自动压缩** —— 若 `rewind` 运行前自动压缩遮蔽了探索的一部分，折叠覆盖表面剩余部分；仅日志的标记在两种情况下都存活。
