<center>
<h1>@dpskh/tool-rewind —— DeepSeek Harness 的探索折叠插件</h1>

[English](README.md) | 中文

[![dshfind](https://dshfind.com/api/card/dpskh/dsh-rewind?lang=zh)](https://dshfind.com/zh/plugins/dpskh/dsh-rewind?ref=badge)

</center>

一个包、一个入口插件。挂载 `@dpskh/tool-rewind` 提供 `ctx.rewind`（把最近一次 checkpoint 标记之后的一切折叠成自动生成报告的服务）和基于它的模型侧 `rewind` 工具。折叠用报告遮蔽探索的表面区间 —— 与 DeepSeek Harness 压缩接缝相同的 `surfaceOp: { op: 'replace' }` 机制 —— 使后续请求不再携带探索的中间步骤（读文件、搜索、试验），而持久日志完整保留探索过程供审计。报告由 `ctx.llm` 自动生成；标记来自兄弟插件 [`@dpskh/tool-checkpoint`](https://github.com/dsh-external/dsh-checkpoint)。不修改任何上游包。

## 配置

```yaml
- id: tool-rewind
  name: '@dpskh/tool-rewind'
  config:
    toolName: rewind                # model-facing tool name (default rewind)
    maxTokens: 1024                 # summarization output budget
    maxSummarizationRetries: 2      # empty-completion retries before failing
    summarizationProvider: deepseek # optional override of the routed provider
    summarizationModel: deepseek-chat
    reportLanguage: en              # en | zh (report instruction language)
```

报告调用的 provider/model 解析顺序：配置覆盖，其次会话最近路由的 `request/header` 配置，最后 agent 的 options。三者全缺则响亮报错。

当路由暴露 `off` reasoning effort 时，报告调用会关闭 thinking —— 压缩探索是机械性任务，而推理模型的思考 token 会在 `maxTokens` 预算内与报告文本竞争（思考可能在任何文本开始前耗尽预算，留下空补全）。空补全最多重试 `maxSummarizationRetries` 次；预算耗尽后以携带 finish 原因与 token 用量的消息拒绝，并体现在拒绝消息中。

## 配合使用

折叠需要标记：两个插件一起挂载，`rewind` 才能折叠 `checkpoint` 锚定的探索。单独挂 `rewind` 没有可折叠的内容，会以 no-checkpoint 错误失败；单独挂 `checkpoint` 只会记录惰性标记。

```yaml
- id: tool-checkpoint
  name: '@dpskh/tool-checkpoint'    # https://github.com/dpskh/dsh-checkpoint
- id: tool-rewind
  name: '@dpskh/tool-rewind'
  config:
    reportLanguage: en              # en | zh report instruction language
```

## 契约

- `ctx.rewind.rewind(agent, signal)` —— 通过 `ctx.checkpoint` 读取会话最近的标记；选取其后的表面区间（绝不折叠 rewind 调用本身，末尾边界绝不切开 tool-call/result 配对）；通过 `ctx.llm` 总结该区间；随后在**一个同步块**内提交携带报告的替换 `user/message`（source 标记 `{ kind: 'plugin', plugin: 'rewind' }`，`isRewindReportSource` 可识别），并通过 `ctx.checkpoint.completeFold` 把标记置为已折叠。失败时**什么都不写** —— 区域原样、标记保持激活、可重试 —— 并以分类 `RewindError`（`NO_CHECKPOINT`、`EMPTY_REGION`、`UNBALANCED`、`FOLD_IN_PROGRESS`、`CHANGED`、`SHRINK`）拒绝。每个会话的进程内锁拒绝并发折叠。调用中途取标记 —— checkpoint 工具运行在它自己的 tool-call 与 result 之间 —— 会在标记后留下孤立的 result；选区恰好跳过该节点，因此 checkpoint 调用自身的一对保持可见，折叠从紧随其后的探索开始。与 checkpoint 并行发出（同一 assistant 消息）的探索调用也会把 result 留在标记之后；这些会被折叠，因此并行探索永远不会坍缩成空区域。
- `rewind` 工具 —— 无参数 → `{ markId, foldedNodes, start, end, foldedChars, reportChars }`，其中 `foldedChars` 是折叠区域贡献的模型可见文本（助手文本、推理、工具参数与工具输出），`reportChars` 是替换报告的字符数 —— 收缩判定数据在每个结果与错误中可见。渲染意图：generic 卡片。
- **回合守卫** —— 当会话存在激活（未折叠）的 checkpoint 标记时，`agent/turn-stopping` 会向 agent 的 next-step 收件箱注入常驻警告（每个 agent 每个回合一次），使回合无法在 `rewind` 折叠探索之前结束。激活状态通过 `ctx.checkpoint` 读取。合理失败的 rewind（如空区域）不会把回合困在警告循环里。

## 模型体验

直接可见：`rewind` 工具调用及其折叠结果。替换后的报告消息从下一次请求起对模型可见；探索的持久事件仍在日志中，但不在模型可见表面。`tool:rewind` 系统提示 section 将折叠纪律变为常驻指令（标记的探索一结束立即 rewind，先于无关工作）。

#### KV 缓存影响

报告调用复用会话的系统提示词与工具 schema，保持为最近一次路由请求的前缀（热前缀缓存）。折叠本身按设计改变后续请求的模型可见前缀。

## 已知限制与待办

- **收缩检查基于字符** —— 报告必须比折叠区域的模型可见文本更短（按字符而非 token 度量）；度量标准统计模型读取的每个 block：助手/用户文本、推理、工具参数与工具输出（递归）。基于 token 计量的检查会更精确；图片 block 没有文本度量标准，保守忽略。
- **无 `off` reasoning effort 路由上的空补全** —— 此类路由保留默认 thinking；若 thinking 耗尽 `maxTokens`，补全为空，只有重试预算能挽救折叠。
- **每个标记只能折叠一次** —— 对同一标记二次折叠会因区域为空而被拒绝；新探索需要新的 `checkpoint`。
- **探索中途发生自动压缩** —— 若 `rewind` 运行前自动压缩遮蔽了探索的一部分，折叠覆盖表面剩余部分；存储中的标记在两种情况下都存活。
