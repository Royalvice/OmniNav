# Contributing

## 开发流程

1. 先阅读贡献文档：
- [REQUIREMENTS.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/REQUIREMENTS.md)
- [IMPLEMENTATION_PLAN.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/IMPLEMENTATION_PLAN.md)
- [TASK.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/TASK.md)
- [WALKTHROUGH.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/WALKTHROUGH.md)

2. 遵循 AGENTS 约束：
- [AGENTS.md](https://github.com/Royalvice/OmniNav/blob/main/AGENTS.md)

## 代码要求

- Batch-first 数据契约 `(B, ...)`
- 跨层类型集中在 `omninav/core/types.py`
- Registry 驱动构建（避免硬编码实例化）
- 关键组件遵循生命周期状态机

## 文档要求

当架构/运行时/接口行为变更时，同步更新：
- Requirements / Plan / Task / Walkthrough / Agents

## 测试

针对改动模块执行单测与集成测试。
重型 Genesis 相关场景按仓库测试开关执行。

参考：
- [tests](https://github.com/Royalvice/OmniNav/tree/main/tests)
