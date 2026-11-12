# fari-rune

视频链接转纯文字：粘贴 B 站 / 微博视频链接，本地转写为人声文本。

- 转写引擎：SenseVoice-Small（阿里开源），中文效果好
- 推理框架：sherpa-onnx（轻量，无需 PyTorch）
- 界面：Gradio 网页（`http://localhost:7860`）
- 隐私：音频不上传，模型与转写全部在本机完成

## 安装（仅首次）

要求 Python 3.10 及以上（用 `python3 --version` 检查）。

```bash
# 1. 安装 ffmpeg（音轨提取）
brew install ffmpeg

# 2. 创建虚拟环境并安装依赖
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## 启动

```bash
venv/bin/python app.py
```

浏览器会自动打开 `http://localhost:7860`。首次启动会下载模型（约 250MB，
放在 `models/` 下），之后完全离线可用。

## 使用

1. 输入框粘贴视频链接（支持 B 站普通/分 P 单集、微博视频）
2. 点「开始转写」，等待进度完成
3. 页面显示纯文本，可复制或下载 TXT（保存在 `outputs/`）
   转写结果按句号分行：每个句号后换行并空一行，便于阅读
4. 左侧「历史记录」列出所有转写结果（标题 / 链接 / 内容预览）：
   - 点击某条历史 → 右侧链接框、状态框、结果框自动回填
   - 「复制」「下载」按钮针对当前选中的历史条目
   - 历史保存在 `outputs/history.json`；旧 `outputs/*.txt` 首次运行时自动导入（无链接信息）

## 性能参考（Apple M1）

| 视频时长 | 转写耗时 |
|---|---|
| 10 分钟 | 约 30~60 秒 |
| 30 分钟 | 约 2~3 分钟 |

## 常见问题

- **下载失败**：确认链接可直接在浏览器打开；B 站限流时可稍后重试。
- **模型下载慢/失败**：手动下载
  [模型包](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)，
  解压到 `models/` 目录。
- **ffmpeg 报错**：确认 `brew install ffmpeg` 已执行。
