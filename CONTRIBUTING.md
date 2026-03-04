# 贡献指南

感谢您对EventGraphRAG项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- 🧪 添加测试用例
- 📊 分享实验结果

## 如何贡献

### 1. 环境设置

在开始贡献之前，请确保您的开发环境已正确设置：

```bash
# 克隆仓库
git clone https://github.com/yourusername/EventGraphRAG.git
cd EventGraphRAG

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install pytest black flake8
```

### 2. 代码规范

我们遵循以下代码规范：

- 使用Black进行代码格式化
- 遵循PEP 8编码规范
- 添加适当的类型注解
- 为新功能编写测试用例

```bash
# 格式化代码
black .

# 检查代码规范
flake8 .

# 运行测试
pytest
```

### 3. 提交流程

1. **Fork仓库**：点击GitHub页面右上角的"Fork"按钮

2. **创建分支**：
```bash
git checkout -b feature/your-feature-name
```

3. **提交更改**：
```bash
git add .
git commit -m "feat: 添加新功能描述"
```

4. **推送到您的fork**：
```bash
git push origin feature/your-feature-name
```

5. **创建Pull Request**：在GitHub上创建PR并描述您的更改

### 4. 提交信息规范

我们使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

- `feat:` 新功能
- `fix:` Bug修复
- `docs:` 文档更新
- `style:` 代码格式化
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建过程或辅助工具的变动

示例：
```
feat: 添加新的事件切分算法
fix: 修复图构建中的内存泄漏问题
docs: 更新README安装说明
```

## 报告Bug

如果您发现Bug，请创建一个Issue并包含以下信息：

- 📋 Bug描述
- 🔄 重现步骤
- 🎯 期望行为
- 🖼️ 实际行为
- 💻 环境信息（操作系统、Python版本等）
- 📎 相关日志或截图

## 功能请求

如果您有新功能建议，请创建一个Issue并描述：

- 💡 功能描述
- 🎯 使用场景
- 🔄 可能的实现方案
- 📊 预期效果

## 开发指南

### 项目结构

```
EventGraphRAG/
├── core/                    # 核心算法模块
│   ├── event_segments.py    # 事件切分
│   ├── graphconstraction.py # 图构建
│   ├── retrival_TOG.py      # TOG检索
│   ├── retrival_others.py   # 其他检索方法
│   ├── answer_rate.py       # 答案生成与评分
│   └── prompt.py           # 提示模板
├── utils/                   # 工具函数
├── data/                    # 数据处理
├── tests/                   # 测试用例（待添加）
└── docs/                    # 文档（待添加）
```

### 添加新功能

1. 在相应模块中实现功能
2. 添加单元测试
3. 更新文档
4. 确保所有测试通过

### 代码审查

所有Pull Request都需要通过代码审查。请确保：

- ✅ 代码遵循项目规范
- ✅ 包含适当的测试
- ✅ 文档已更新
- ✅ 没有引入新的Bug
- ✅ 性能没有显著下降

## 实验结果分享

我们欢迎您分享使用EventGraphRAG的实验结果：

- 📊 性能对比实验
- 🔍 消融实验结果
- 🎯 新应用场景
- 💡 改进建议

请通过Issue或Pull Request分享您的发现。

## 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

## 联系方式

如有任何问题，请通过以下方式联系我们：

- 📧 邮箱：your.email@example.com
- 🐛 GitHub Issues
- 💬 讨论区（待添加）

---

感谢您的贡献！🎉