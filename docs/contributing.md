# 贡献指南

感谢你对 OmniNav 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告 Bug

1. 在 [GitHub Issues](https://github.com/Royalvice/OmniNav/issues) 中搜索是否已有类似问题
2. 如果没有，创建新 Issue，包含：
   - 清晰的问题描述
   - 复现步骤
   - 预期行为 vs 实际行为
   - 环境信息 (OS, Python 版本, GPU 等)

### 提交代码

1. Fork 本仓库
2. 创建功能分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -m "feat: add your feature"`
4. 推送分支: `git push origin feature/your-feature`
5. 创建 Pull Request

### 代码规范

- 使用 [Black](https://github.com/psf/black) 格式化代码
- 使用 [isort](https://pycqa.github.io/isort/) 排序 import
- 遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- 为新功能添加单元测试

## 开发环境设置

```bash
# 克隆你的 fork
git clone https://github.com/YOUR_USERNAME/OmniNav.git
cd OmniNav

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/
```

## 文档贡献

文档使用 MkDocs + Material 主题编写，位于 `docs/` 目录。

```bash
# 安装文档依赖
pip install mkdocs-material mkdocstrings[python]

# 本地预览
mkdocs serve
```

## 许可证

通过贡献代码，你同意你的贡献将按照项目的 Apache-2.0 许可证进行许可。
