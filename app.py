"""fari-rune — 视频链接转纯文字

流程：粘贴 B 站 / 微博视频链接 → yt-dlp 提取音轨 → SenseVoice 本地转写 → 输出纯文本。
全程本地运行，音频不上传任何服务器。
"""

import re
import shutil
import subprocess
import tarfile
import time
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import yt_dlp

# imageio-ffmpeg 在系统无 ffmpeg 时提供静态二进制（延迟导入，见 get_ffmpeg）

# ---------- 常量 ----------

ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp"      # 临时音频文件
OUT_DIR = ROOT / "outputs"  # 转写结果 TXT
MODEL_DIR = ROOT / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
)
SAMPLE_RATE = 16000         # 模型要求 16kHz
CHUNK_SECONDS = 30          # 转写分块时长（长视频分块，防内存暴涨）
SEARCH_WINDOW_SEC = 2       # 切块时在 ±2 秒内找停顿处下刀
DOWNLOAD_RETRIES = 3        # 模型下载重试次数
TAG_RE = re.compile(r"<\|[^|]*\|>")  # SenseVoice 输出的语言/情感标记

recognizer = None  # 全局单例，见 load_recognizer()


# ---------- 模型 ----------

def ensure_model() -> Path:
    """模型缺失时下载并解压，返回模型目录。失败自动重试，损坏包不残留。"""
    model_file = MODEL_DIR / "model.int8.onnx"
    if model_file.exists():
        return MODEL_DIR

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = MODEL_DIR.parent / MODEL_URL.rsplit("/", 1)[-1]
    print(f"首次运行：下载模型（约 250MB）\n{MODEL_URL}")
    try:
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                _download_archive(archive)
                with tarfile.open(archive, "r:bz2") as tar:  # 解压同时校验完整性
                    tar.extractall(MODEL_DIR.parent)
                break
            except (requests.RequestException, tarfile.TarError, OSError) as e:
                print(f"\n第 {attempt}/{DOWNLOAD_RETRIES} 次尝试失败：{e}")
                archive.unlink(missing_ok=True)
                if attempt == DOWNLOAD_RETRIES:
                    raise
                time.sleep(5 * attempt)
    finally:
        archive.unlink(missing_ok=True)
    print("模型就绪。")
    return MODEL_DIR


def _download_archive(archive: Path) -> None:
    """流式下载模型压缩包，带进度打印。"""
    with requests.get(MODEL_URL, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r下载中 {done / total:.0%}", end="", flush=True)
    print()


def load_recognizer():
    """加载 SenseVoice 模型（进程内单例）。"""
    global recognizer
    if recognizer is not None:
        return recognizer
    import sherpa_onnx

    model_dir = ensure_model()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        use_itn=True,
    )
    print("模型加载完成。")
    return recognizer


# ---------- 下载与预处理 ----------

def download_audio(url: str, dest_dir: Path) -> tuple[Path, str]:
    """下载视频音轨（不下载画面），返回 (音频文件, 标题)。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,   # B 站分 P 视频只取当前这一 P
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = Path(ydl.prepare_filename(info))
    title = info.get("title") or filepath.stem
    return filepath, title


def get_ffmpeg() -> str:
    """优先用系统 ffmpeg，否则用 imageio-ffmpeg 自带的静态二进制。"""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_to_wav(src: Path) -> Path:
    """ffmpeg 转成 16kHz 单声道 wav。"""
    dst = src.with_suffix(".wav")
    subprocess.run(
        [
            get_ffmpeg(), "-y",
            "-i", str(src),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst


def read_wav(path: Path) -> np.ndarray:
    """读取 wav，返回 [-1, 1] 的 float32 波形。"""
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SAMPLE_RATE:
            raise ValueError(f"采样率异常：{f.getframerate()}")
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


# ---------- 转写 ----------

def split_chunks(samples: np.ndarray, sr: int) -> list[np.ndarray]:
    """把整段音频按 CHUNK_SECONDS 切块，尽量切在停顿（能量最低）处。"""
    target = CHUNK_SECONDS * sr
    window = SEARCH_WINDOW_SEC * sr
    probe = sr // 10  # 100ms 一格，用于估算能量
    chunks: list[np.ndarray] = []
    start = 0
    while len(samples) - start > target * 1.5:
        mid = start + target
        lo = max(start + target // 2, mid - window)
        hi = min(len(samples), mid + window)
        rms = [
            np.sqrt(np.mean(samples[i : i + probe] ** 2))
            for i in range(lo, hi, probe)
        ]
        cut = lo + int(np.argmin(rms)) * probe
        chunks.append(samples[start:cut])
        start = cut
    if start < len(samples):
        chunks.append(samples[start:])
    return chunks


def decode_chunk(rec, samples: np.ndarray) -> str:
    """单块音频解码，返回清理标记后的文本。"""
    stream = rec.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    rec.decode_stream(stream)
    return TAG_RE.sub("", stream.result.text).strip()


def format_text(text: str) -> str:
    """按句号排版：每个句号后换行并空一行，便于阅读。"""
    return re.sub(r"。+", "。\n\n", text).strip()


def sanitize_filename(name: str) -> str:
    """去掉文件系统不允许的字符。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()[:80] or "result"


# ---------- 主流程 ----------

def run(url: str, progress=gr.Progress()):
    """Gradio 主流程：下载 → 转码 → 分块转写 → 输出 TXT。"""
    url = (url or "").strip()
    if not url:
        yield "请先粘贴视频链接。", "", None
        return

    try:
        rec = load_recognizer()
        yield "①/4 模型就绪，正在下载音频 ...", "", None
        audio, title = download_audio(url, TMP_DIR)

        yield "②/4 提取音轨中 ...", "", None
        wav = audio_to_wav(audio)
        samples = read_wav(wav)

        chunks = split_chunks(samples, SAMPLE_RATE)
        texts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            texts.append(decode_chunk(rec, chunk))
            desc = f"③/4 转写中 第 {i}/{len(chunks)} 块"
            progress(i / len(chunks), desc=desc)
            yield desc, "", None

        text = format_text("".join(texts))
        out_path = OUT_DIR / f"{sanitize_filename(title)}.txt"
        out_path.write_text(text, encoding="utf-8")
        yield f"④/4 完成：《{title}》", text, str(out_path)
    except Exception as e:  # noqa: BLE001 — 界面上直接展示失败原因
        yield f"失败：{e}", "", None
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# ---------- 界面 ----------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="fari-rune 视频转文字") as demo:
        gr.Markdown(
            "## fari-rune\n"
            "粘贴 B 站 / 微博视频链接，本地转写为纯文本。"
        )
        url = gr.Textbox(
            label="视频链接",
            placeholder="https://www.bilibili.com/video/... 或微博视频链接",
        )
        btn = gr.Button("开始转写", variant="primary")
        status = gr.Textbox(label="状态", interactive=False)
        output = gr.Textbox(label="转写结果", lines=16)
        copy_btn = gr.Button("📋 复制全文")
        copy_btn.click(
            fn=None,
            inputs=[output],
            outputs=None,
            # 纯前端复制：剪贴板只在 localhost（安全上下文）可用，本项目满足
            js="(text) => { if (text) navigator.clipboard.writeText(text); }",
        )
        file = gr.File(label="下载 TXT")
        btn.click(run, inputs=url, outputs=[status, output, file])
    return demo


if __name__ == "__main__":
    TMP_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    load_recognizer()  # 启动即加载，首次转写不用再等
    build_demo().queue(max_size=8).launch(inbrowser=True, show_error=True)
