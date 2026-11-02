# 斯坦福 CS224N 第四讲《句法结构与依存句法分析》电子书预览指南

本目录包含由视频 `https://www.bilibili.com/video/BV1U5RNYgEfp/?p=4` 完整转写并深度重构生成的图文技术电子书。

---

## 快速启动预览（3 种方式）

### 方式 1：双击一键脚本（推荐 ⭐）

直接双击运行本目录下的脚本：
```cmd
preview.bat
```
脚本会自动启动本地 HTTP 服务，并在您的默认浏览器中自动打开网页：
👉 **http://localhost:8082/book.html**

---

### 方式 2：终端命令行启动

在本目录（`D:\learnbook\cs224n\lecture4`）打开 PowerShell 或命令提示符，执行以下命令：

```powershell
python -m http.server 8082
```

> **说明**：如果系统未配置全局 Python，可直接调用主项目的虚拟环境：
> ```powershell
> d:\project\videobook\.venv\Scripts\python.exe -m http.server 8082
> ```

终端显示 `Serving HTTP on :: port 8082` 后，在浏览器中访问：
👉 **http://localhost:8082/book.html**

---

### 方式 3：更换端口启动（若 8082 被占用）

如果您的 8082 端口被其他软件占用，可以指定其他空闲端口（例如 8090）：

```powershell
python -m http.server 8090
```
然后在浏览器访问：
👉 **http://localhost:8090/book.html**

---

## 为什么必须通过 HTTP 服务访问？

> ⚠️ **请勿直接双击 `book.html` 用浏览器本地文件模式（`file:///...`）打开！**
> 
> 原因：
> 1. **CORS 跨域与 Cookie 隐私隔离**：电子书内嵌了 B 站原画卡片与交互视频组件，浏览器对 `file://` 协议有极严格的跨域安全限制，直接双击会导致播放器拒载或黑屏。
> 2. **KaTeX & Mermaid 资源加载**：数学公式排版（KaTeX/MathJax）与矢量流程图引擎在本地 HTTP 服务（`http://`）环境下能保证 100% 稳定解析与渲染。

---

## 如何停止服务？

在运行服务的控制台黑窗口中按下 `Ctrl + C`，或者直接关闭控制台窗口即可安全退出。
