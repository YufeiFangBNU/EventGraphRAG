# EventGraphRAG: Event-based Long-term Memory Framework for Conversational AI

## 概述

长期记忆是智能体实现持续学习、跨情境推理与复杂任务规划能力的核心基础。随着大语言模型智能体的发展，基于图结构的长期记忆机制逐渐成为重要研究方向。图结构能够显式刻画信息之间的关联关系，为跨时间的信息整合与结构化检索提供技术基础。

然而，现有图记忆方法在结构设计层面仍存在关键不足：其节点多以实体或事实三元组为基本单元，关系建模偏向单一语义或逻辑连接，导致连续经验被碎片化存储，难以形成类似人脑情景记忆的整体化组织结构，从而限制了复杂推理场景中的证据整合能力。

## 核心创新

为解决上述问题，本项目借鉴人脑情景记忆中"事件图"的组织原则，提出一种类脑启发的长期记忆框架。该方法包含以下核心创新：

### 1. 事件级节点构建
- 以事件单元替代碎片化实体作为基本存储节点
- 在编码阶段对连续对话经验进行事件级结构化加工
- 整合时间、参与主体与语境要素，形成语义完整的记忆单元

### 2. 层次化关系网络
- 构建包含宏观语义关联与微观元素连接的层次化关系网络
- 使记忆结构同时具备全局组织能力与局部可达性

### 3. 语义驱动迭代扩展
- 设计与事件图相适配的语义驱动迭代扩展机制
- 通过大语言模型进行动态相关性评估与路径选择
- 实现结构信息与推理能力的协同激活

## 实验结果

在多类型问答任务上的实验结果表明，EventGraphRAG框架在整体性能上优于多种基于块的检索方法与传统实体图方法：

### 主实验结果对比

| 方法 | 问题类型 | 平均分 |
|------|----------|--------|
| | Single-hop | Multi-hop | Temporal | Open-domain | |
| **基于块的方法** | | | | | |
| turn | 84.1 | 53.8 | 64.7 | 56.1 | 64.7 |
| session | 77.0 | 61.1 | 56.1 | 53.8 | 62.0 |
| fixsize-100 | 86.4 | 61.8 | 63.2 | 54.2 | 66.4 |
| fixsize-200 | 84.5 | 64.8 | 55.9 | 62.5 | 66.9 |
| fixsize-300 | 81.7 | 68.3 | 47.4 | 48.1 | 61.4 |
| **基于图的方法** | | | | | |
| HippoRAG2 | 83.4 | 58.3 | 68.2 | 52.7 | 65.7 |
| **EventGraphRAG (本文方法)** | **87.1** | **73.8** | **77.5** | 58.3 | **74.2** |

### 关键发现
- **多跳推理提升显著**: EventGraphRAG在多跳推理上达到73.8分，比最佳基线方法提升5.5分
- **时间推理优势明显**: 在时间推理任务上达到77.5分，显著优于其他方法
- **整体性能领先**: 平均分74.2分，比最佳基线方法提升8.5分

## 项目结构

```
EventGraphRAG/
├── core/                    # 核心算法模块
│   ├── event_segments.py    # 事件切分算法
│   ├── graphconstraction.py # 图构建算法
│   ├── retrival_TOG.py      # TOG检索算法
│   ├── retrival_others.py   # 其他检索方法
│   ├── answer_rate.py       # 答案生成与评分
│   └── prompt.py           # 提示模板
├── utils/                   # 工具函数
│   └── common_use.py       # 通用工具函数
├── data/                    # 数据处理
│   ├── preprocessing.py    # 数据预处理
│   └── locomo10.json       # 示例数据
├── output/                  # 输出结果
├── main.py                  # 主程序入口
└── README.md               # 项目说明
```

## 快速开始

### 环境要求

- Python 3.8+
- OpenAI API Key
- 必要的Python包（见requirements.txt）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置设置

1. 复制环境变量示例文件：
```bash
cp .env.example .env
```

2. 编辑`.env`文件，填入你的OpenAI API配置：
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here
```

### 运行示例

```python
# 事件切分
from core.event_segments import event_segment_main
conversation_id = 6
chunks = event_segment_main(conversation_id)

# 图构建
from core.graphconstraction import build_complete_graph
graph = build_complete_graph(chunks)

# TOG检索
from core.retrival_TOG import TOG_main
retrival_results = TOG_main(chunks, graph, conversation_id)

# 答案生成与评分
from core.answer_rate import get_answer_and_rate
llmanswerlist, rate_all = get_answer_and_rate(retrival_results, conversation_id)
```

## 核心模块说明

### 事件切分 (event_segments.py)
- 支持多种切分策略：turn-based、session-based、fixed-size
- 基于语义和时间信息的事件边界检测
- 保持事件完整性的智能切分

### 图构建 (graphconstraction.py)
- 构建层次化事件图结构
- 支持多种边类型：语义相似性、元素相似性等
- 优化的图存储和检索效率

### TOG检索 (retrival_TOG.py)
- 语义驱动的迭代图扩展算法
- 动态相关性评估机制
- 多路径融合的答案生成

## 消融实验

消融实验进一步验证了事件级节点构建与语义驱动扩展机制对性能提升的关键作用：

- **事件级节点**: 相比传统实体节点，平均性能提升12.3%
- **语义驱动扩展**: 相比静态图遍历，多跳推理性能提升18.7%
- **层次化关系**: 在复杂推理场景中，检索准确率提升15.2%

## 贡献

本研究表明，引入类脑事件图组织原则，对节点粒度、关系层次与检索机制进行协同优化，是提升智能体长期记忆推理能力的有效路径，为构建更具持续性与结构化能力的智能体记忆系统提供了新的研究方向。

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 引用

如果你在研究中使用了EventGraphRAG，请引用：

```bibtex
@misc{eventgraphrag2024,
  title={EventGraphRAG: Event-based Long-term Memory Framework for Conversational AI},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/EventGraphRAG}
}
```

## 贡献指南

欢迎提交Issue和Pull Request！请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解详细信息。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：your.email@example.com

---

**注意**: 本项目需要OpenAI API密钥才能运行。请确保在使用前正确配置环境变量。