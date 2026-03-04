# 图片说明

本目录包含项目文档中使用的图片文件。

## 主实验结果图

### main_results.png
- **描述**: EventGraphRAG与各基线方法在不同问题类型上的性能对比图
- **包含内容**: 
  - 单跳问题 (Single-hop) 对比
  - 多跳问题 (Multi-hop) 对比  
  - 时间推理问题 (Temporal) 对比
  - 开放域问题 (Open-domain) 对比
- **生成方式**: 基于论文表2数据使用matplotlib生成
- **尺寸**: 建议1200x800像素，清晰显示各方法差异

## 如何生成图片

如果需要重新生成实验结果图片，可以使用以下代码：

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据来自论文表2
methods = ['turn', 'session', 'fixsize-100', 'fixsize-200', 'fixsize-300', 'HippoRAG2', 'EventGraphRAG']
single_hop = [84.1, 77.0, 86.4, 84.5, 81.7, 83.4, 87.1]
multi_hop = [53.8, 61.1, 61.8, 64.8, 68.3, 58.3, 73.8]
temporal = [64.7, 56.1, 63.2, 55.9, 47.4, 68.2, 77.5]
open_domain = [56.1, 53.8, 54.2, 62.5, 48.1, 52.7, 58.3]

# 设置图表
x = np.arange(len(methods))
width = 0.15

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x - 3*width/2, single_hop, width, label='Single-hop', color='#1f77b4')
rects2 = ax.bar(x - width/2, multi_hop, width, label='Multi-hop', color='#ff7f0e')
rects3 = ax.bar(x + width/2, temporal, width, label='Temporal', color='#2ca02c')
rects4 = ax.bar(x + 3*width/2, open_domain, width, label='Open-domain', color='#d62728')

ax.set_ylabel('分数')
ax.set_title('EventGraphRAG与基线方法性能对比')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('main_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 注意事项

1. 图片应保持学术风格，使用清晰的颜色对比
2. 确保文字可读性，避免重叠
3. 保持数据准确性，与论文中的表格数据一致
4. 建议使用高分辨率输出 (300dpi)