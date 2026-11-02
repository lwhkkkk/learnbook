# 个人技术课程图文电子书库 (LearnBook)

本项目归档由高质量公开课、视频教程及官方原版资料重构而成的中英文双语图文电子书。

每个课程讲次均包含：
- 完整重构的教科书级 Markdown 讲义（含严谨 LaTeX 数学公式推导、Mermaid 架构与计算流程图）
- 支持本地独立启动的交互式 HTML 电子书（带 KaTeX/MathJax 数学渲染、图表与视频原画截图）
- 官方课件（Slides）与官方讲义（Notes）PDF 原件及文本归档
- 精校行级时间戳字幕与 ASR 对照表

---

## 📚 已收录课程目录

### 1. 斯坦福 CS224N：深度学习自然语言处理 (Winter 2024)

- **Lecture 2**: [词向量进阶、多义性与神经网络分类器](cs224n/lecture2/README.md)
  - **主题**：Word2Vec 负采样推导、共现矩阵与 SVD 降维、GloVe 对数双线性模型、词向量内部/外部评估、一词多义与 Arora 线性叠加定理、前馈神经网络分类器。
  - **讲义源码**：[`cs224n/lecture2/book.md`](cs224n/lecture2/book.md)
  - **网页交互版**：[`cs224n/lecture2/book.html`](cs224n/lecture2/book.html)
  - **快捷预览**：进入 `cs224n/lecture2` 目录双击 `preview.bat` 即可启动阅读。

- **Lecture 3**: [神经网络与反向传播算法（手算矩阵微积分与计算图）](cs224n/lecture3/README.md)
  - **主题**：人工神经元与 NER 上下文窗口分类、非线性激活函数对比（Sigmoid/tanh/ReLU）、矩阵微积分与雅可比矩阵（Jacobian）、手算单层网络参数与输入梯度（误差向量 $\delta$ 与外积法则）、计算图与反向传播单节点动力学、加法/乘法/Max 计算门直觉分析、模块化 Forward/Backward API 与 PyTorch Autograd 原理、数值梯度检验（Gradient Checking）。
  - **讲义源码**：[`cs224n/lecture3/book.md`](cs224n/lecture3/book.md)
  - **网页交互版**：[`cs224n/lecture3/book.html`](cs224n/lecture3/book.html)
  - **快捷预览**：进入 `cs224n/lecture3` 目录双击 `preview.bat` 即可启动阅读。
