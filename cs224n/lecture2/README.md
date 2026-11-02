# 斯坦福 CS224N 第二讲《词向量与语言模型》电子书预览指南

本目录包含由视频 `https://www.bilibili.com/video/BV1U5RNYgEfp/?p=2` 完整转写并重构生成的图文技术电子书。

---

## 快速启动预览（3 种方式）

### 方式 1：双击一键脚本（推荐 ⭐）

返回项目根目录 `d:\project\videobook\`，直接双击运行：
```
preview.bat
```
脚本会自动启动本地 HTTP 服务，并在您的默认浏览器中自动打开网页。

---

### 方式 2：终端命令行启动

在项目根目录打开 PowerShell 或 VS Code 终端，执行以下命令：

```powershell
.venv\Scripts\python.exe -m http.server 8080 --directory output\BV1U5RNYgEfp_p2
```

> **说明**：如果当前系统环境变量中已有可用 Python，也可简写为：
> ```bash
> python -m http.server 8080 --directory output/BV1U5RNYgEfp_p2
> ```

终端显示 `Serving HTTP on :: port 8080` 后，在浏览器打开：
👉 **http://localhost:8080/book.html**

---

### 方式 3：更换端口启动（若 8080 被占用）

如果您的 8080 端口被其他软件（如 Tomcat/Vue 开发服务）占用，可以指定其他端口（如 8088）：

```powershell
.venv\Scripts\python.exe -m http.server 8088 --directory output\BV1U5RNYgEfp_p2
```
然后在浏览器访问：
👉 **http://localhost:8088/book.html**

---

## 为什么必须通过 HTTP 服务访问？

> ⚠️ **请勿直接双击 `book.html` 用浏览器本地文件模式（`file:///...`）打开！**
> 
> 原因：
> 1. **CORS 跨域与 Cookie 隐私隔离**：电子书内嵌了 B 站高清原画卡片与视频播放组件，浏览器对 `file://` 协议有极严格的跨域安全限制，直接双击会导致播放器拒载或黑屏。
> 2. **KaTeX & Mermaid 资源加载**：数学公式排版与矢量流程图引擎在本地 HTTP 服务（`http://`）环境下能保证 100% 稳定解析与渲染。

---

## 如何停止服务？

在运行该命令的黑窗口（终端）中，直接按下键盘上的 **`Ctrl + C`**，或者直接关闭该终端窗口即可退出服务。

---

## 目录文件清单

| 文件名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`book.html`** | 核心成品 | 渲染完成的交互式 HTML 电子书（公式/图表/视频卡片） |
| **`book.md`** | 电子书源码 | 结构化 Markdown 讲义，含公式推导与图表代码 |
| **`transcript.corrected.txt`** | 对照稿 | 经 AI 行级校对的带时间戳字幕逐行对照稿 |
| **`transcript.json`** | 原始字幕 | Whisper large-v3 产出的高精度时间戳字幕数据（99.8% 覆盖率） |
| **`images/`** | 截图目录 | 从视频中提取的 17 张高清纯净关键帧截图 |
| **`course/`** | 官方资料 | 斯坦福官方课件 `slides.pdf` 与讲义 `notes.pdf` 归档 |
