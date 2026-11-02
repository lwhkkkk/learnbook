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
  - **快捷预览**：进入 `cs224n/lecture2` 目录双击 `preview.bat` 即可启动阅读（端口 8080）。

- **Lecture 3**: [神经网络与反向传播算法（手算矩阵微积分与计算图）](cs224n/lecture3/README.md)
  - **主题**：人工神经元与 NER 上下文窗口分类、非线性激活函数对比（Sigmoid/tanh/ReLU）、矩阵微积分与雅可比矩阵（Jacobian）、手算单层网络参数与输入梯度（误差向量 $\delta$ 与外积法则）、计算图与反向传播单节点动力学、加法/乘法/Max 计算门直觉分析、模块化 Forward/Backward API 与 PyTorch Autograd 原理、数值梯度检验（Gradient Checking）。
  - **讲义源码**：[`cs224n/lecture3/book.md`](cs224n/lecture3/book.md)
  - **网页交互版**：[`cs224n/lecture3/book.html`](cs224n/lecture3/book.html)
  - **快捷预览**：进入 `cs224n/lecture3` 目录双击 `preview.bat` 即可启动阅读（端口 8081）。

- **Lecture 4**: [句法结构与依存句法分析（从传统转移系统到神经依存模型）](cs224n/lecture4/README.md)
  - **主题**：短语结构文法（Constituency）vs 依存文法（Dependency）、自然语言歧义性（介词短语依附歧义、卡塔兰数 $C_n$ 组合爆炸、并列范围歧义）、依存树理论基础（支配词 Head、从属词 Dependent、虚拟根节点 ROOT、投射性 Projectivity 与交叉弧）、基于转移的移进-归约状态机（Arc-Standard 转移系统：Shift / Left-Arc / Right-Arc）、MaltParser 传统特征工程瓶颈、标准评测指标（UAS 与 LAS）、陈丹琦与 Manning 经典神经依存解析器（Chen & Manning 2014：稠密分布式嵌入、三次方激活函数 $f(z)=z^3$、千句/秒吞吐）、基于图的依存解析（最大生成树 MST、Chu-Liu-Edmonds 算法与 Deep Biaffine Parser）。
  - **讲义源码**：[`cs224n/lecture4/book.md`](cs224n/lecture4/book.md)
  - **网页交互版**：[`cs224n/lecture4/book.html`](cs224n/lecture4/book.html)
  - **快捷预览**：进入 `cs224n/lecture4` 目录双击 `preview.bat` 即可启动阅读（端口 8082）。
