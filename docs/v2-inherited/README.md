# v2 继承文档（M0 迁入）

本目录是从 `Echo-Smith/mirror-memory-v2`（已归档，只读）迁入的文档，作为
v1 吸收 v2 能力（M0–M5 合并计划）的需求与设计基线：

| 文件 | 内容 | 对应里程碑 |
|---|---|---|
| [requirements.md](requirements.md) | 需求契约 R01–R15 | 全部（验收依据） |
| [invariants.md](invariants.md) | 6 条不变量 I1–I6 | M1（I1/I3/I4/I6）、M3（I5）、M4（I2） |
| [budget-guardrails.md](budget-guardrails.md) | 预算护栏设计 | M1 / M5 可观测性 |
| [decisions/](decisions/) | v2 决策记录（契约冻结、三阶段 Worker、预算护栏） | 参考依据 |

说明：

- 文档内出现的 v2 相对链接（如 `validation.md`、`../../docs/plans/...`）
  指向 v2 仓库内的文件，v1 仓库不含这些文件；需要时到归档仓库查阅。
- 文档正文中「Mirror / v2 / M1-M2」等术语指 v2 的实现阶段，阅读时注意与
  v1 合并计划里的里程碑编号（M0–M5）区分。
- 需求契约与不变量在 v1 语境下的落地方式以 Notion「合并计划：mirror v1
  吸收 v2 能力」页面为准。
