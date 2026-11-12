# AGENTS.md — fari-rune 开发指南

视频链接转纯文字：B 站 / 微博链接 → yt-dlp 提取音轨 → SenseVoice 本地转写 → 纯文本。

## 结构

- `app.py` — 唯一源文件，单文件应用，按注释分区：
  - 常量 / 模型（ensure_model、load_recognizer）/ 下载与预处理 / 转写
  - 历史记录（load_history、save_history、history_to_samples 等）
  - 主流程 run()（生成器，yield 5 元组：状态、结果、文件、历史列表、历史数据）
  - 界面 build_demo()（左侧历史目录 + 右侧转写区）
- `models/` — SenseVoice 模型（gitignore，首次运行自动下载）
- `outputs/` — 转写结果 TXT + `history.json` 历史索引（gitignore）
  - history.json 每条：`{url, title, text, time}`；旧 TXT 无 url，首次运行导入
  - 同一 URL 重复转写会覆盖旧条目并置顶
- `run.log` — 运行日志（gitignore，含访问过的 URL）

## 命令

```bash
venv/bin/python app.py   # 启动，浏览器自动打开 http://localhost:7860
```

## 约定

- 中文注释，注释解释“为什么”而非“是什么”
- Gradio 6.x：界面交互用 gr.Dataset(type="index") + gr.State 传递历史数据
- 剪贴板复制用纯前端 JS（localhost 安全上下文），不走后端
- 转写过程全程本地，音频不上传
