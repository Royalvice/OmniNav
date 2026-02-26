# Contributing（贡献指南）

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

文档质量门槛：
- Sphinx 构建必须 `0 warning`。
- docs 中的 GitHub `blob/main` 链接必须指向真实文件。
- 中英文文档树保持镜像结构。
- 术语与标题风格通过文档样式检查。

## 测试

针对改动模块执行单测与集成测试。
重型 Genesis 相关场景按仓库测试开关执行。

参考：
- [tests](https://github.com/Royalvice/OmniNav/tree/main/tests)

## 文档校验命令

```bash
python scripts/docs/check_repo_links.py
python scripts/docs/check_bilingual_structure.py
python scripts/docs/check_docs_style.py
source ~/omninav_ros_env/bin/activate && sphinx-build -b html docs docs/_build/html -W --keep-going
```
