"""fari-rune — 视频链接转纯文字

流程：粘贴 B 站 / 微博视频链接 → yt-dlp 提取音轨 → SenseVoice 本地转写 → 输出纯文本。
全程本地运行，音频不上传任何服务器。
"""

import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import wave
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import yt_dlp

# imageio-ffmpeg 在系统无 ffmpeg 时提供静态二进制（延迟导入，见 get_ffmpeg）

# ---------- 常量 ----------

ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp"              # 临时音频文件
OUT_DIR = ROOT / "outputs"          # 转写结果 TXT
HISTORY_FILE = OUT_DIR / "history.json"  # 历史记录（链接 + 全文）
TIME_FMT = "%Y-%m-%d %H:%M"   # 历史时间戳格式（字符串排序即时间排序）
PREVIEW_CHARS = 60            # 历史列表里的文字预览长度
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
                _download_archive(MODEL_URL, archive)
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


def _download_archive(url: str, archive: Path) -> None:
    """流式下载压缩包到 archive，带进度打印。"""
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
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


# ---------- 界面字体 ----------

FONTS_DIR = ROOT / "fonts"  # MiSans woff2 存放目录（gitignore，缺失时自动下载）
FONTS_URL = "https://hyperos.mi.com/font-download/MiSans.zip"  # 小米官方免费商用字体
FONT_FILES_REQUIRED = ("MiSans-Regular.woff2", "MiSans-Medium.woff2")  # 必备两档：400 正文 / 500 强调
FONT_FILES_OPTIONAL = ("MiSans-Demibold.woff2",)  # 可选：600 标题字重，官方包缺失时回退 500
ZIP_INNER_DIR = "MiSans/woff2/"  # 官方压缩包内 woff2 所在路径


def ensure_fonts() -> None:
    """界面字体缺失时从小米官网下载（约 217MB，仅一次）。只提取所需 woff2，不整包解压。

    与 ensure_model 不同：字体失败只回退系统字体，不影响转写，故不抛异常阻断启动。
    可选字重（600）缺失只跳过该文件并回退 500；判断“是否已就绪”只看必备字重，
    否则可选字重缺失会导致每次启动都重新下载整包。
    """
    if all((FONTS_DIR / name).exists() for name in FONT_FILES_REQUIRED):
        return

    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    archive = FONTS_DIR / FONTS_URL.rsplit("/", 1)[-1]
    print(f"首次运行：下载 MiSans 界面字体（约 217MB）\n{FONTS_URL}")
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            _download_archive(FONTS_URL, archive)
            with zipfile.ZipFile(archive) as zf:
                # 可选字重缺文件只跳过，不进入重试
                for name in FONT_FILES_OPTIONAL:
                    try:
                        with zf.open(ZIP_INNER_DIR + name) as src, open(FONTS_DIR / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    except KeyError:
                        print(f"官方字体包缺少 {name}，标题 600 字重将回退为 500。")
                missing = [n for n in FONT_FILES_REQUIRED if not (FONTS_DIR / n).exists()]
                if missing:
                    raise KeyError(f"字体包内缺少必备字重：{missing}")
            print("字体就绪。")
            return
        except (requests.RequestException, zipfile.BadZipFile, KeyError, OSError) as e:
            print(f"\n第 {attempt}/{DOWNLOAD_RETRIES} 次尝试失败：{e}")
            archive.unlink(missing_ok=True)
            if attempt == DOWNLOAD_RETRIES:
                print("字体下载失败，界面回退系统字体。")
                return
            time.sleep(5 * attempt)


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


# ---------- 历史记录 ----------

# 只用于统计/分享跳转的链接参数：归一化时剔除，同一视频的不同分享写法即可命中同一条历史
TRACKING_PARAMS = {
    # B 站：vd_source / spm_id_from 等来自页内分享与数据统计
    "spm", "spm_id_from", "from_spm_id", "vd_source", "buvid", "seid", "scm",
    "launch_id", "unique_k", "share_source", "share_medium", "share_platfrom",
    "share_tag", "share_session_id", "share_token",
    # 微博：转发落地页带的各种来源标记
    "mod", "fr", "refer_flag",
    # 通用统计参数
    "from", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
}


def normalize_url(url: str) -> str:
    """链接归一化：同一视频的不同写法归为同一个键。

    剔除 TRACKING_PARAMS 与结尾斜杠，保留其余参数——尤其是 B 站的 p
    （分 P，不同 P 是不同内容，绝不能合并）；p=1 是默认第一 P，与不带 p 等价。
    解析失败的串原样返回。
    """
    url = url.strip()
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return url
    query = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if k not in TRACKING_PARAMS and not (k == "p" and v in ("", "1"))
    ]
    return urllib.parse.urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/") or "/",
            urllib.parse.urlencode(query),
            "",
        )
    )


def dedupe_history(entries: list[dict]) -> list[dict]:
    """同一链接只保留最新一条（列表新→旧），保证历史目录里链接不重合。

    旧文件导入的条目没有链接（url 为空），不参与去重。
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for e in entries:
        url = e["url"]
        if url:
            key = normalize_url(url)
            if key in seen:
                continue
            seen.add(key)
        unique.append(e)
    return unique


def load_history() -> list[dict]:
    """读取历史记录（新→旧）。首次运行时把旧 outputs/*.txt 导入（无链接信息）。"""
    if HISTORY_FILE.exists():
        try:
            return dedupe_history(json.loads(HISTORY_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            print("history.json 损坏，重新从 outputs/*.txt 导入。")

    entries: list[dict] = []
    for txt in OUT_DIR.glob("*.txt"):
        try:
            entries.append({
                "url": "",
                "title": txt.stem,
                "text": txt.read_text(encoding="utf-8", errors="replace"),
                "time": time.strftime(TIME_FMT, time.localtime(txt.stat().st_mtime)),
            })
        except OSError:
            continue
    entries.sort(key=lambda e: e["time"], reverse=True)
    entries = dedupe_history(entries)
    save_history(entries)
    return entries


def save_history(entries: list[dict]) -> None:
    """历史记录写入磁盘。"""
    HISTORY_FILE.write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )


def export_entry_txt(e: dict) -> Path:
    """把历史条目导出为 TXT（outputs/ 下缺失时补写），返回文件路径。

    历史侧不再放独立下载按钮，点历史条目回填右侧时直接激活右下角下载。
    """
    path = OUT_DIR / f"{sanitize_filename(e['title'])}.txt"
    if not path.exists():
        path.write_text(e["text"], encoding="utf-8")
    return path


def history_to_samples(entries: list[dict]) -> list[list[str]]:
    """把历史记录转成 Dataset 行：标题 / 内容预览。

    链接列对辨认内容毫无帮助（长 URL 纯噪音），时间列冗余
    （目录本来就按时间倒序），都砍掉，只留能辨认条目的两列。
    """
    rows = []
    for e in entries:
        preview = re.sub(r"\s+", " ", e["text"]).strip()[:PREVIEW_CHARS]
        rows.append([e["title"], preview])
    return rows


# ---------- 版本管理（outputs 自动 commit） ----------

GIT_TIMEOUT = 30  # 单条 git 命令超时（秒）：卡死也不能拖住转写主流程


def _git(*args: str) -> subprocess.CompletedProcess:
    """在 outputs/ 嵌套仓库里执行一条 git 命令，捕获输出，调用方按返回码处理。"""
    return subprocess.run(
        ["git", *args], cwd=OUT_DIR, capture_output=True, text=True, timeout=GIT_TIMEOUT
    )


def ensure_git_repo() -> None:
    """outputs/ 独立版本库：首次自动 init。

    用嵌套仓库而非并入主仓库：转写内容（个人记录）与代码历史隔离，主仓库
    推远端不会带出转录文本。排除项写 .git/info/exclude 而非 outputs/.gitignore：
    同样全局生效，且不在 outputs/ 里落多余文件。提交者身份缺失不在这里兜底
    编造，由 auto_commit() 报错让用户自己配置。
    """
    if not (OUT_DIR / ".git").exists():
        _git("init")
    # history.json 每次全量重写（n 条记录转写 n 次就复制 n 份），入库会让仓库
    # 线性膨胀；.DS_Store 是 Finder 元数据，主仓库的 .gitignore 管不到嵌套仓库
    exclude = OUT_DIR / ".git" / "info" / "exclude"
    if exclude.exists() and "history.json" not in exclude.read_text(encoding="utf-8"):
        with exclude.open("a", encoding="utf-8") as f:
            f.write("\n# 派生索引与系统文件，不入库\nhistory.json\n.DS_Store\n")


def auto_commit(verb: str, title: str) -> str | None:
    """outputs/ 有变化就提交一条：动词 + 标题（转录 / 更新 / 补回）。

    返回 None 或错误文案，由调用方上状态行红字报错（三个触发点都有状态行）。
    提交者身份缺失不在服务端编造兑底——没有真实身份就没有可信的提交记录，
    必须报错让用户自己 git config。无变化（重复转写同一视频且内容一致）
    静默跳过；纯本地仓库，禁止 push。
    """
    try:
        ensure_git_repo()
        if not _git("config", "user.name").stdout.strip() or not _git(
            "config", "user.email"
        ).stdout.strip():
            return (
                "outputs 提交失败：git 提交者身份未配置，"
                "请先执行 git config --global user.name / user.email"
            )
        _git("add", "-A")
        # --quiet：有暂存改动返回 1，无变化返回 0，0 即无事可提交
        if _git("diff", "--cached", "--quiet").returncode == 0:
            return None
        result = _git("commit", "-m", f"{verb} {title}")
        if result.returncode == 0:
            print(f"outputs 已提交：{verb} {title}")
            return None
        return f"outputs 提交失败：{result.stderr.strip()}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"outputs 自动提交不可用：{e}"


# ---------- 主流程 ----------

def _status(text: str, error: bool = False):
    """状态行输出：error=True 时附加 status-error 类，CSS 据此把文字转绛红。

    必须用组件实例而非裸字符串：字符串只更新 value，不会清掉上一次的
    error 类，失败红字会粘在后续成功状态上。
    """
    return gr.Markdown(
        value=text,
        elem_id="status-text",
        elem_classes=["status-error"] if error else [],
    )


def run(url: str, progress=gr.Progress()):
    """Gradio 主流程：下载 → 转码 → 分块转写 → 输出 TXT，并写入历史。

    yield 7 元组：状态、结果、下载文件、目录更新、历史数据、主按钮、复制按钮。
    复制按钮跟随结果框：开始转写即清空结果 → 禁用；复用历史/完成 → 激活。
    """
    url = (url or "").strip()
    if not url:
        yield (
            _status("请先粘贴视频链接。", error=True),
            "", None,
            gr.skip(), gr.skip(),
            gr.Button(value="开始转写", interactive=True),
            gr.Button(value="复制全文", interactive=False),
        )
        return

    # 主按钮忙碌态：点下立即禁用并换成“转写中…”，防止转写期间重复提交，结束时恢复。
    # 组件实例作为输出即“按构造参数更新该组件”（同下方 DownloadButton 用法）
    busy_btn = gr.Button(value="转写中…", interactive=False)
    idle_btn = gr.Button(value="开始转写", interactive=True)
    no_copy = gr.Button(value="复制全文", interactive=False)
    ok_copy = gr.Button(value="复制全文", interactive=True)
    yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), busy_btn, no_copy

    # 同一视频（含不同分享参数的写法）已在历史中：直接复用旧结果，不再下载转写
    key = normalize_url(url)
    history = load_history()
    if existing := next(
        (e for e in history if e["url"] and normalize_url(e["url"]) == key), None
    ):
        # TXT 被手动删过就从历史补写，保证下载可用；补写也是 outputs 变化，入版本库
        restored = not (OUT_DIR / f"{sanitize_filename(existing['title'])}.txt").exists()
        out_path = export_entry_txt(existing)
        git_error = auto_commit("补回", existing["title"]) if restored else None
        yield (
            _status(
                f"已转写过：《{existing['title']}》（{existing['time']}），直接复用历史结果"
                + (f"；{git_error}" if git_error else ""),
                error=bool(git_error),
            ),
            existing["text"],
            gr.DownloadButton(value=str(out_path), interactive=True),
            gr.skip(),  # 历史无变化，目录不动
            gr.skip(),
            idle_btn,
            ok_copy,
        )
        return

    try:
        rec = load_recognizer()
        yield _status("①/④ 正在下载音频 ..."), "", None, gr.skip(), gr.skip(), busy_btn, no_copy
        audio, title = download_audio(url, TMP_DIR)

        yield _status("②/④ 正在提取音轨 ..."), "", None, gr.skip(), gr.skip(), busy_btn, no_copy
        wav = audio_to_wav(audio)
        waveform = read_wav(wav)

        chunks = split_chunks(waveform, SAMPLE_RATE)
        texts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            texts.append(decode_chunk(rec, chunk))
            desc = f"③/④ 转写中 第 {i}/{len(chunks)} 块"
            progress(i / len(chunks), desc=desc)
            yield _status(desc), "", None, gr.skip(), gr.skip(), busy_btn, no_copy

        text = format_text("".join(texts))
        out_path = OUT_DIR / f"{sanitize_filename(title)}.txt"
        # 写前判定新旧：同名文件已在 = 重转同一视频（覆盖），commit 动词据此区分
        verb = "更新" if out_path.exists() else "转录"
        out_path.write_text(text, encoding="utf-8")

        entry = {"url": url, "title": title, "text": text, "time": time.strftime(TIME_FMT)}
        # 同链接（归一化后）只保留最新一条；url 为空的旧导入条目原样保留
        history = [e for e in history if not e["url"] or normalize_url(e["url"]) != key]
        history.insert(0, entry)
        save_history(history)
        # 结果落盘后同步入 outputs 版本库；提交失败不改写主结果，只在完成状态里报错
        git_error = auto_commit(verb, title)
        # Gradio 6 的 Dataset 收到裸样本列表只会"查找"不会"替换"（触发警告且返回 None），
        # 必须回传新的 Dataset 对象，左侧目录才会立即显示新条目
        # 下载按钮初始 interactive=False，完成后才挂上文件并启用；
        # 过程中 yield None 会清掉上一次的下载文件，与原 gr.File 行为一致
        yield (
            _status(
                f"④/④ 完成：《{title}》" + (f"；{git_error}" if git_error else ""),
                error=bool(git_error),
            ),
            text,
            gr.DownloadButton(value=str(out_path), interactive=True),
            gr.Dataset(samples=history_to_samples(history)),
            history,
            idle_btn,
            ok_copy,
        )
    except Exception as e:  # noqa: BLE001 — 界面上直接展示失败原因
        yield _status(f"失败：{e}", error=True), "", None, gr.skip(), gr.skip(), idle_btn, no_copy
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# ---------- 界面 ----------

def refresh_history():
    """页面加载（含刷新）时重读磁盘历史。

    build_demo 只在进程启动时执行一次，目录初始数据是启动时刻的快照；
    不在页面加载时重读的话，刷新浏览器也看不到之后新转写的条目。
    """
    history = load_history()
    return (
        gr.Dataset(samples=history_to_samples(history)),
        history,
    )


def on_history_select(index: int, history: list[dict]):
    """点击历史条目：回填链接 / 状态 / 转写结果，并激活右下角下载与复制。"""
    if not 0 <= index < len(history):
        return "", "", "", gr.DownloadButton(interactive=False), gr.Button(interactive=False)
    e = history[index]
    time_note = f"（{e['time']}）" if e["time"] else ""
    # TXT 被手动删过时 export 会补写；补写也是 outputs 变化，同样入版本库
    restored = not (OUT_DIR / f"{sanitize_filename(e['title'])}.txt").exists()
    path = export_entry_txt(e)
    git_error = auto_commit("补回", e["title"]) if restored else None
    return (
        e["url"],
        f"已转写：《{e['title']}》{time_note}" + (f"；{git_error}" if git_error else ""),
        e["text"],
        gr.DownloadButton(value=str(path), interactive=True),
        gr.Button(interactive=True),
    )


# 前端视觉 = SHELL_CSS（:root 设计 token + 页壳几何）+ CUSTOM_CSS（.contain 内部
# 规格实现）+ build_theme()（token 引用 :root 变量）。改色只改 :root，勿写死字面值。

# 运行记录彻底关闭：Gradio 6 的 run_history 开关在 launch() 参数上，但
# `gradio app.py` 开发模式的 CLI 自己调 launch，带不上我们的参数（同
# footer_links 不走 launch 的原因），环境变量才是两种启动模式都生效的通道。
# False = 浏览器不再记录运行、已存记录被前端清除、/gradio_api/runs 页 404、
# 设置弹层对应分区不渲染（CSS 另有兜底）
os.environ["GRADIO_RUN_HISTORY"] = "false"

# 把 fonts/、assets/ 暴露为静态目录，经 /file= 路由引用（字体文件 / favicon）
gr.set_static_paths([str(ROOT / "fonts"), str(ROOT / "assets")])

# favicon：SVG 经 head= 注入 <link>，与 SHELL_CSS 同通道——launch(favicon_path)
# 在 `gradio app.py` 开发模式下带不上（CLI 自己调 launch），head= 才能两模式都生效。
# 路由用 /gradio_api/file=（Gradio 6.x 静态文件正式前缀，curl 实测 200）；根级
# /file= 在 6.30 由前端 service worker 兑底，favicon 是浏览器原生请求不保险
FAVICON_HTML = (
    '<link rel="icon" type="image/svg+xml" href="/gradio_api/file=assets/icon.svg" />'
)

# 设置弹层锁死：可配置项（显示主题 / 语言 / 渐进式 Web 应用 / Screen Studio /
# 运行记录）全部删除，弹层只留标题栏。分区隐藏的 CSS 见 SHELL_CSS「设置弹层精简」，
# 运行记录另有 GRADIO_RUN_HISTORY=false 环境变量从服务端整体关闭（见常量区）。以下两件必须在 Gradio
# 前端模块脚本启动前以内联脚本定死（head= 内联脚本先于模块脚本执行，两种启动模式
# 都生效）：
# ① 主题锁 Light：Gradio 主题 = ?__theme= URL 参数，缺省跟随系统深浅色。启动前
#    replaceState 补写 __theme=light（只改本页地址不刷新）；切换 UI 已随分区删除，
#    URL 参数即唯一事实源，手动输 __theme=dark 也会被改写回 light。
# ② 语言锁中文：Gradio 前端初始语言取 navigator.language（svelte-i18n，无持久化
#    键，设置里的语言选择也不落盘），覆写为 zh-CN 走内置中文语言包（弹层标题
#    「设置」等框架文案）。
SETTINGS_LOCK_HTML = r"""
<script>
(function () {
  try {
    var u = new URL(window.location.href);
    if (u.searchParams.get('__theme') !== 'light') {
      u.searchParams.set('__theme', 'light');
      window.history.replaceState(null, '', u.toString());
    }
  } catch (e) { /* 地址不可写时退回跟随系统；:root 视觉 token 本身恒为浅色 */ }
  try {
    Object.defineProperty(navigator, 'language', { configurable: true, get: function () { return 'zh-CN'; } });
    Object.defineProperty(navigator, 'languages', { configurable: true, get: function () { return ['zh-CN']; } });
  } catch (e) { /* 个别环境不允许覆写时，语言跟随浏览器设置 */ }
})();
</script>
"""

# 页壳样式 + 设计 token：必须走 head= 注入的原生 <style>。
# 为什么不放 CUSTOM_CSS：Gradio 前端会把 css= 里的每条选择器强制改写成
# `.gradio-container... .contain <选择器>`（prefix_css），只能命中 .contain
# 内部的元素；而根容器 / main / wrap / contain / footer 都在 .contain 外，
# 写在 css= 里根本够不着。:root token 同理必须放这里，否则 :root 会被
# prefix_css 改写成 `.contain :root` 而失效（在 css= 里用 var() 引用则没问题）。
SHELL_CSS = """
<style>
/* ===== 设计 token（改色只改这里；全部蓝系衍生，禁纯黑与无色相纯灰）===== */
:root {
  /* 背景表面 */
  --bg-canvas: #F4F7FB;          /* 全局底层画布：极淡冰川灰蓝 */
  --bg-surface: #FFFFFF;         /* 核心工作表面：纯白（仅输入框、结果框） */
  --bg-subtle: #EBF1F8;          /* 次级表面：淡雾蓝（左栏、次级按钮底） */
  --bg-hover: #E2ECF7;           /* 悬停态背景：浅海沫蓝 */
  --bg-active: #D3E4F8;          /* 选中态背景：清澈淡蓝 */

  /* 文本颜色（全部混入深海蓝） */
  --text-primary: #0F233C;       /* 主文本/正文：深海蓝黑 */
  --text-secondary: #435B78;     /* 次要文本/小节标题：岩石蓝灰 */
  --text-muted: #7C93B0;         /* 占位符/预览/空态引导：冷雾蓝灰 */
  --text-status: #1E64B4;        /* 状态行细蓝字：冷静知性蓝 */

  /* 品牌与强调色（唯一能量源） */
  --accent-primary: #1A5FB4;     /* 主按钮实色、选中指示条：沉稳钴蓝 */
  --accent-hover: #144B8F;       /* 主按钮悬停：深邃海蓝 */
  --accent-disabled: #8FAFD8;    /* 主按钮忙碌/禁用：低饱和灰蓝 */
  --accent-focus-ring: rgba(26, 95, 180, 0.16); /* 聚焦光晕 */

  /* 边框与分割线 */
  --border-subtle: #DCE6F2;      /* 默认极细边框 */
  --border-focus: #2B72C9;       /* 输入框聚焦边框 */

  /* 语义微调色（保持在蓝/冷调逻辑内） */
  --status-error: #C93B5B;       /* 失败原因：冷调绛红 */
  --status-success: #0E7C7B;     /* 已复制/完成微调：深青蓝 */
}

/* ---- 单屏锁死（桌面端 >768px）：整页 100vh，滚动交给左右两栏内部 ---- */
html, body { height: 100%; margin: 0; background: var(--bg-canvas); }
.gradio-container { height: 100vh !important; overflow: hidden !important; background: var(--bg-canvas); color: var(--text-primary); }
.gradio-container > .main { flex: 1 1 0 !important; min-height: 0 !important; overflow: hidden !important; }
.gradio-container > .main > .wrap { flex: 1 1 0 !important; min-height: 0 !important; overflow: hidden !important; padding: 0 !important; }
/* 居中容器：1152px + 左右 24px；.fillable 断点规则特异性更高，必须 !important 压过 */
.gradio-container main.contain {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    height: 100%;
    width: 100% !important;  /* 显式满宽：交叉轴 auto-margin 会把 flex item 收缩为 fit-content */
    max-width: 1152px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}
/* Gradio 6 会把顶层组件包进一个匿名 column（contain 的直接孩子），三段式的
   flex 链必须显式撑满它；css= 里写 main.contain > .column 会被 prefix_css
   改写成 `.contain main.contain > .column`（自己不是自己的后代）而失效，
   所以这两组规则必须在 head= 通道 */
.gradio-container main.contain > .column {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}
.gradio-container main.contain > .column > .block { margin: 0 !important; padding: 0 !important; }
.gradio-container main.contain > .column > .block .html-container { padding: 0 !important; }

/* ---- Footer：原生设置按钮只出生在 footer（无头诊断确认的 DOM 事实），已被
   SHELL_JS 整颗搬进顶栏（.app-topbar）行最右；余下 API / Gradio / 运行记录全是噪音，
   footer 空了即整体隐藏（连带收回原 44px 条带，单屏可用高度 +44px）---- */
.gradio-container footer[aria-label="Gradio footer navigation"] { display: none !important; }

/* ---- 设置弹层精简：Gradio 自带弹层（div.api-docs-wrap）只留标题栏 ---- */
/* 弹层内容 = 一排 .banner-wrap：标题栏（唯一含 h2 .url 的一个）/ 显示主题 /
   语言 / 渐进式 Web 应用 / Screen Studio beta / 运行记录。标题栏以外整体隐藏；
   标题旁的应用地址文字（div.url）一并隐藏。主题与语言的锁死见
   SETTINGS_LOCK_HTML；运行记录另由 GRADIO_RUN_HISTORY=false 环境变量关闭
   （分区直接不渲染，这里是兜底）。弹层在 .contain 外，必须走 head= 通道 */
.api-docs-wrap .banner-wrap:not(:has(h2 .url)) { display: none !important; }
.api-docs-wrap h2 .url { display: none !important; }

/* ---- 窄窗口 ≤768px：解除单屏，恢复整页滚动（组件区规则见 CUSTOM_CSS）---- */
@media (max-width: 768px) {
    html, body { height: auto; overflow: visible; }
    .gradio-container { height: auto !important; min-height: 100vh; overflow: visible !important; }
    .gradio-container > .main,
    .gradio-container > .main > .wrap {
        flex: 0 0 auto !important;
        overflow: visible !important;
    }
    .gradio-container main.contain { flex: 0 0 auto !important; height: auto; }
}
</style>
"""

# 页壳行为脚本：与 SHELL_CSS 同通道走 head=（两种模式都生效）。Gradio 只在 footer
# 里生成原生设置按钮（button.settings），Svelte 挂载晚于脚本执行，故用
# MutationObserver 兜底重查；发现后把整颗按钮（「设置」文字 + 齿轮 img）搬到顶栏
# .app-topbar 末尾（品牌标题 + 副标题所在行的最右）。行内已有设置按钮时跳过，
# 防 footer 重渲染产生重复条目
SHELL_JS = """
<script>
(function () {
    function relocate() {
        const footer = document.querySelector('footer[aria-label="Gradio footer navigation"]');
        const btn = footer && footer.querySelector('button.settings');
        const bar = document.querySelector('.app-topbar');
        if (!btn || !bar || bar.querySelector('button.settings')) return;
        bar.appendChild(btn);
    }
    new MutationObserver(relocate).observe(document.documentElement, { childList: true, subtree: true });
    relocate();
})();
</script>
"""

CUSTOM_CSS = """
/* fari-rune 前端视觉 —— 「冷静蓝调」设计规格实现（仅 .contain 内部）。
 * 设计 token（:root）与页壳几何在 SHELL_CSS；改色一律改 :root 变量，勿写字面值。
 * 字阶：品牌标题 20/600 · 副标题 13 · 小节标题 12/600/0.06em ·
 *       正文（输入框/结果）15/1.75 · 列表与按钮 13/500 · 状态行 13/行高24（条高 34：文字 24 + 底部 10 滚动条带）
 *       （进度/时间/百分比 tabular-nums 防数字抖动）
 */

/* MiSans 本地字体：400 正文 / 500 强调 / 600 标题三档（600 为可选下载，缺失回退 500）；
   font-display swap 保证加载前先出系统字不白屏 */
@font-face {
    font-family: "MiSans";
    src: url("/file=fonts/MiSans-Regular.woff2") format("woff2");
    font-weight: 400;
    font-display: swap;
}
@font-face {
    font-family: "MiSans";
    src: url("/file=fonts/MiSans-Medium.woff2") format("woff2");
    font-weight: 500;
    font-display: swap;
}
@font-face {
    font-family: "MiSans";
    src: url("/file=fonts/MiSans-Demibold.woff2") format("woff2");
    font-weight: 600;
    font-display: swap;
}

/* ---- Header（64px）：标题 + 副标题垂直居中，底边框 1px ---- */
.app-topbar {
    height: 64px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0;
    border-bottom: 1px solid var(--border-subtle);
}
.app-topbar .logo { width: 24px; height: 24px; flex-shrink: 0; display: block; margin-right: -4px; }
.app-topbar .title { font-size: 20px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.2; color: var(--text-primary); }
.app-topbar .sub { font-size: 13px; font-weight: 400; line-height: 1.4; color: var(--text-muted); }

/* 设置按钮：原生 footer 按钮经 SHELL_JS 整颗搬到顶栏行最右。「设置」13/500 冷雾蓝灰，
   悬停转深海蓝黑；margin-left:auto 顶到行右缘——桌面端副标题不定宽，靠它才贴最右
   （≤768px 时 .sub 已 flex:1，auto 无副作用） */
.app-topbar > button.settings {
    flex-shrink: 0;
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0 4px;
    background: transparent;
    border: none;
    box-shadow: none;
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: color 150ms ease;
}
.app-topbar > button.settings:hover { color: var(--text-primary); }
/* 齿轮是 data:svg 的 <img>（硬编码 #808080 灰，无头诊断确认）：CSS color 染不动，
   只能调尺寸 / 悬停加深 / 旋转；svg 选择器命中不了它 */
.app-topbar > button.settings img {
    width: 18px;
    height: 18px;
    margin: 0;                  /* 归零 Gradio 默认左右 4px 边距，间距改由 gap 控制 */
    transition: transform 150ms ease, filter 150ms ease;
}
.app-topbar > button.settings:hover img {
    transform: rotate(30deg);
    filter: brightness(0.4);    /* 悬停加深灰齿轮，与深色文字协调 */
}

/* ---- Main：左栏 280px + 右栏弹性 / gap 24 / 上下 20px（三段式的核心区）----
   Gradio 6 DOM 事实（无头诊断确认）：.row 默认 align-items:flex-start 不拉伸子项；
   Column 带 inline flex-grow 与 min-width:320px；故这里全部显式 + !important */
#app-row {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 24px !important;
    padding: 20px 0 !important;
    border: none !important;
    background: transparent !important;
}
/* 左栏：Gradio 的 inline flex-grow:1 / min-width:320px 必须用 !important 压掉 */
#history-col {
    flex: 0 0 280px !important;
    min-width: 0 !important;
    height: auto !important;
    overflow: hidden;
    display: flex !important;
    flex-direction: column !important;
    background: var(--bg-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px 0 !important;
}
/* 左栏里「历史」小标题的 block/html 包装去默认边距 */
#history-col > .block { margin: 0 !important; padding: 0 !important; }
/* 小节标题：与「转写结果」同级对仗（主从呼应） */
.history-head { padding: 0 16px 10px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; line-height: 1.5; color: var(--text-secondary); }

/* Dataset：自身透明（容器已着色），逐级撑满后让表格区自滚 */
#history-ds {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    display: flex;
    flex-direction: column;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    --table-odd-background-fill: transparent;
    --table-even-background-fill: transparent;
    --table-row-focus: var(--bg-hover);
}
#history-ds > .wrap { flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; padding: 0; }
/* 列表滚动区：左右 8px；极细滚动条默认半透明，悬停变深 */
#history-ds .table-wrap {
    flex: 1 1 0;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 0 8px;
    scrollbar-width: thin;
    scrollbar-color: #BDD0E7 transparent;
}
#history-ds .table-wrap::-webkit-scrollbar { width: 4px; }
#history-ds .table-wrap::-webkit-scrollbar-track { background: transparent; }
#history-ds .table-wrap::-webkit-scrollbar-thumb { background: rgba(189, 208, 231, 0.55); border-radius: 2px; }
#history-ds .table-wrap::-webkit-scrollbar-thumb:hover { background: #BDD0E7; }

/* 条目行：去表头；border-spacing 0 2px 模拟 2px 条目间距（table 行不支持 margin） */
#history-ds .tr-head { display: none; }
#history-ds table { table-layout: fixed; width: 100%; border-collapse: separate; border-spacing: 0 2px; background: transparent; }
#history-ds td { height: 38px; padding: 0 12px !important; border: none; background: transparent; text-align: left; }
#history-ds td > span { display: block; overflow: hidden; }
#history-ds td, #history-ds td * { white-space: nowrap; text-overflow: ellipsis; }
/* 两列 45%/55%：左标题 13/500 主文本，右预览 12 muted 右对齐 */
#history-ds td:nth-child(1) { width: 45%; }
#history-ds td:nth-child(2) { width: 55%; text-align: right; }
#history-ds td:nth-child(1), #history-ds td:nth-child(1) * { font-size: 13px; font-weight: 500; color: var(--text-primary); }
#history-ds td:nth-child(2), #history-ds td:nth-child(2) * { font-size: 12px; color: var(--text-muted); }
/* 悬停/选中：背景打在 td 上才能出圆角（tr 的 border-radius 不生效） */
#history-ds tr:not([aria-selected="true"]):hover td { background: var(--bg-hover); }
#history-ds tr:not([aria-selected="true"]):hover td:first-child { border-radius: 6px 0 0 6px; }
#history-ds tr:not([aria-selected="true"]):hover td:last-child { border-radius: 0 6px 6px 0; }
#history-ds tr[aria-selected="true"] td { background: var(--bg-active); }
#history-ds tr[aria-selected="true"] td:first-child { border-radius: 6px 0 0 6px; position: relative; }
#history-ds tr[aria-selected="true"] td:last-child { border-radius: 0 6px 6px 0; }
/* 选中指示蓝条：3x18 圆角 2px，距条目左缘内 2px，垂直居中 */
#history-ds tr[aria-selected="true"] td:first-child::before {
    content: "";
    position: absolute;
    left: 2px;
    top: 50%;
    transform: translateY(-50%);
    width: 3px;
    height: 18px;
    border-radius: 2px;
    background: var(--accent-primary);
}
/* 选中时标题加深为品牌蓝 */
#history-ds tr[aria-selected="true"] td:first-child,
#history-ds tr[aria-selected="true"] td:first-child * { color: var(--accent-primary); }
/* 空态引导：tbody 无条目时滚动区整体 flex 居中（:has 不支持则退化为顶部，无害） */
#history-ds .table-wrap:has(tbody:empty) { display: flex; align-items: center; justify-content: center; }
#history-ds tbody:empty::after {
    content: "转写完成后，记录会出现在这里";
    display: block;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
}

/* ---- 右栏：转写工作区 ---- */
#trans-col {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    height: auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}
/* 右栏直接孩子的 Gradio block/form 一律去默认边距，间距统一由 gap 12px 控制 */
#trans-col > .block, #trans-col > .form {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* 5.1 输入行：42px，链接框吃掉剩余宽度，回车即转写（url.submit） */
#input-row {
    height: 42px !important;
    flex: 0 0 auto !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 12px !important;
    width: 100% !important;
}
/* 链接框：Gradio 6 以 textarea 渲染（即便单行）且 JS 会写 inline height，
   故 height/overflow 必须 !important；white-space:pre 禁软换行，超长横向滚动。
   .form 自带 Gradio 默认蓝边框（与聚焦边框叠成“宽边框”）；#url-input 的
   padded 内边距会把 textarea 顶出容器 10px（底边被 overflow-y:hidden 裁掉）。
   两者都显式去除，聚焦态才是规格的 1px 边框 + 四周完整 3px 光晕 */
#input-row .form {
    flex: 1 1 0 !important;
    height: 42px !important;
    min-height: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
#url-input {
    height: 42px !important;
    min-height: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}
#url-input textarea {
    height: 42px !important;
    min-height: 42px !important;
    padding: 0 14px !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
    line-height: 40px !important;
    white-space: pre !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    resize: none !important;
    box-shadow: none !important;
    scrollbar-width: thin;
    scrollbar-color: #BDD0E7 transparent;
}
#url-input textarea::placeholder { color: var(--text-muted) !important; }
#url-input textarea:focus {
    outline: none !important;
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-focus-ring) !important;
}

/* 主按钮：全屏唯一视觉重心。elem_id 直接落在 <button> 本身（Gradio 6），
   颜色必须写在这里：dev 模式 CLI 不注入 theme，theme token 不可靠 */
#run-btn {
    height: 42px !important;
    flex: 0 0 auto !important;
    padding: 0 22px !important;
    background: var(--accent-primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: all 150ms ease;
    box-shadow: none !important;
}
#run-btn:hover { background: var(--accent-hover) !important; }
#run-btn:disabled { background: var(--accent-disabled) !important; cursor: not-allowed; }

/* 5.2 状态行：条高 34px（24px 行高 + 底部 10px 滚动条带），固定占位防空态/有态跳动；
   失败红字由 _status() 切 status-error 类。
   超宽文字横向滚动看全文，滚动条与链接框同款 thin 蓝灰——省略号方案不可行（无头诊断确认）：
   Gradio 的 markdown-wrapper 不收缩，「…」截在视口外看不见，右侧还留出一段空白无效滚动区。
   块自带内联 overflow:auto（此处 !important 显式接管），滚动条与文字的间隙由多出的 10px 提供；
   p 溢出保持 visible，行框才能撑出块的滚动宽度，滚到右侧能看到完整文字 */
#status-text {
    height: 34px !important;
    flex: 0 0 auto !important;
    display: flex !important;
    align-items: center;
    margin: 0 !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin;
    scrollbar-color: #BDD0E7 transparent;
}
#trans-col > #status-text { padding: 0 4px !important; }
#status-text p {
    color: var(--text-status);
    font-size: 13px;
    font-weight: 400;
    line-height: 24px;
    font-variant-numeric: tabular-nums;
    margin: 0;
    white-space: nowrap;
    overflow: visible;
}
#status-text.status-error p { color: var(--status-error); }

/* 5.3 结果框：白底容器 + 框内顶栏（36px 小节标题）+ 正文自滚。
   外层还有 Gradio 的 .form 包装，必须先撑满右栏剩余高度 */
#trans-col > .form {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}
#result-box {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    height: auto !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(15, 35, 60, 0.02);
    padding: 0 !important;
}
#result-box label.container {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    height: auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}
/* 框内顶栏：36px「转写结果」小节标题 + 底部分割线。
   Gradio 6 没有 .label-wrap，label 的直接孩子是 span.block-info */
#result-box label > span[data-testid="block-info"] {
    flex: 0 0 auto !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    padding: 0 18px !important;
    margin: 0 !important;
    border-bottom: 1px solid var(--bg-subtle) !important;
    background: transparent !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    line-height: 1.5 !important;
    color: var(--text-secondary) !important;
}
/* 正文滚动区：18/20 内距，15px/1.75 长文阅读体验，极细滚动条。
   JS 会写 inline height/overflow-y（如 height:456px），全部 !important 压制 */
#result-box .input-container {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}
#result-box .input-container > textarea {
    flex: 1 1 0 !important;
    height: 100% !important;
    min-height: 0 !important;
    resize: none !important;
    border: none !important;
    background: var(--bg-surface) !important;
    padding: 18px 20px !important;
    font-size: 15px !important;
    line-height: 1.75 !important;
    color: var(--text-primary) !important;
    overflow-y: auto !important;
    box-shadow: none !important;
    scrollbar-width: thin;
    scrollbar-color: #BDD0E7 transparent;
}
#result-box .input-container > textarea:focus { outline: none !important; box-shadow: none !important; }
#result-box .input-container > textarea::placeholder { color: var(--text-muted) !important; user-select: none; }
#result-box .input-container > textarea::-webkit-scrollbar { width: 4px; }
#result-box .input-container > textarea::-webkit-scrollbar-track { background: transparent; }
#result-box .input-container > textarea::-webkit-scrollbar-thumb { background: rgba(189, 208, 231, 0.55); border-radius: 2px; }
#result-box .input-container > textarea::-webkit-scrollbar-thumb:hover { background: #BDD0E7; }

/* 5.4 操作行：36px；复制靠左、下载最右（稳定的底边锚点）。
   两个次级按钮的 elem_id 也在 button 本身（下载按钮无 id，用 last-child） */
#actions-row {
    height: 36px !important;
    flex: 0 0 auto !important;
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    align-items: center !important;
    gap: 12px !important;
    width: 100% !important;
    margin: 0 !important;
}
/* 次级按钮通用样式：绝不抢主按钮风头 */
#actions-row button {
    height: 34px !important;
    flex: 0 0 auto !important;
    padding: 0 14px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    background: var(--bg-subtle) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: none !important;
    min-width: 88px !important;
    text-align: center;
    transition: all 150ms ease;
    margin: 0 !important;
}
#actions-row button:hover:not(:disabled) {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-color: #BDD0E7 !important;
}
#actions-row button:disabled { opacity: 0.45 !important; cursor: not-allowed !important; }
/* 「已复制」反馈：青蓝微调（JS 切 .copied 类，1.5s 后移除恢复） */
#copy-btn.copied {
    background: #E6F4F4 !important;
    color: var(--status-success) !important;
    border-color: var(--status-success) !important;
    opacity: 1 !important;
}

/* ---- 窄窗口 ≤768px：解除单屏，竖排堆叠（转写区在上、目录在下）---- */
@media (max-width: 768px) {
    #app-row { flex-direction: column !important; gap: 20px !important; flex: 0 0 auto !important; }
    #trans-col { order: 1; flex: 0 0 auto !important; }
    #history-col { order: 2; flex: 0 0 100% !important; max-height: 260px; }
    #trans-col > .form { flex: 0 0 auto !important; }
    #result-box { flex: 0 0 auto !important; height: 60vh; min-height: 320px; }
    #history-ds { flex: 0 0 auto !important; }
    #history-ds .table-wrap { max-height: 200px; }
}
/* 超窄屏 <480px：副标题单行省略 */
@media (max-width: 480px) {
    .app-topbar .sub { flex: 1 1 0; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
}
"""


def build_theme() -> gr.themes.Base:
    """主题只负责字体与色相基底；颜色与几何一律由 :root token + CSS 控制。

    原因（无头诊断确认）：`gradio app.py` 开发模式的 CLI 自己调 launch，
    Blocks 上的 theme 不会被注入前端，只有普通模式生效；两模式视觉必须一致，
    所以 CSS 是唯一视觉来源（theme 即便注入也与 CSS 同源不冲突）。
    """
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.blue,
        # MiSans 本地字体（@font-face 见 CUSTOM_CSS），规格系统栈兑底；基类 Font 不触发 Google Fonts 请求
        font=[gr.themes.Font("MiSans"), "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
    )


def build_demo() -> gr.Blocks:
    history = load_history()

    # Gradio 6 把 theme/css 挪到了 launch()，但那会丢掉 `gradio app.py`
    # 开发模式的热重载主题（CLI 自己调 launch，带不上我们的参数），
    # 所以仍传给 Blocks（当前版本仅打一条弃用警告，功能完整）
    with gr.Blocks(title="fari-rune 视频转文字", theme=build_theme(), css=CUSTOM_CSS, head=SHELL_CSS + SHELL_JS + FAVICON_HTML + SETTINGS_LOCK_HTML) as demo:
        gr.HTML(
            "<div class='app-topbar'>"
            "<img class='logo' src='/gradio_api/file=assets/icon.svg' alt='' />"
            "<span class='title'>fari-rune</span>"
            "<span class='sub'>粘贴 B 站 / 微博视频链接，本地转写为纯文本</span>"
            "</div>"
        )

        # Main 核心区：CSS Grid 280px + 1fr（页壳把本 Row 撑成三段式的 Main）
        with gr.Row(elem_id="app-row"):
            # ----- 左栏：历史目录（从）-----
            # 一套操作只存在一处：点条目回填右侧，复制/下载统一走右下角，
            # 左栏不放任何按钮，避免用户理解两套按钮
            with gr.Column(scale=1, elem_id="history-col"):
                # 左栏唯一文字：「历史」小节标题，与右栏「转写结果」label 对仗（主从呼应）
                gr.HTML("<div class='history-head'>历史</div>")
                history_state = gr.State(history)  # 与历史列表同源的条目数据
                history_ds = gr.Dataset(
                    headers=["标题", "预览"],
                    components=["textbox", "textbox"],
                    samples=history_to_samples(history),
                    type="index",  # 点击回调直接拿到行索引
                    samples_per_page=1000,  # 全部展示，不分页
                    show_label=False,  # 组件默认 "Examples" 标签是噪音，目录自身就是标识
                    elem_id="history-ds",
                )

            # ----- 右栏：转写工作区（主）-----
            with gr.Column(scale=2, elem_id="trans-col"):
                with gr.Row(elem_id="input-row"):
                    # 无 label：占位文案直接说明用法；autofocus 页面加载即聚焦
                    url = gr.Textbox(
                        show_label=False,
                        placeholder="粘贴 B 站 / 微博视频链接，回车直接转写",
                        autofocus=True,
                        elem_id="url-input",
                    )
                    btn = gr.Button("开始转写", variant="primary", scale=0, elem_id="run-btn")
                # 状态行：固定 24px 高占位防跳动；失败红字由 _status() 切 status-error 类
                status = gr.Markdown(elem_id="status-text")
                output = gr.Textbox(
                    label="转写结果",
                    lines=16,
                    placeholder="转写完成后，全文会显示在这里",
                    elem_id="result-box",
                )
                with gr.Row(elem_id="actions-row"):
                    # 复制按钮跟随结果框：无结果禁用，完成/回填历史时由 run() 激活
                    copy_btn = gr.Button("复制全文", size="sm", scale=0, interactive=False, elem_id="copy-btn")
                    file = gr.DownloadButton("下载 TXT", size="sm", interactive=False, scale=0)

        # 点击历史：回填链接框、状态行、结果框，下载与复制按钮随条目激活
        history_ds.click(
            on_history_select,
            inputs=[history_ds, history_state],
            outputs=[url, status, output, file, copy_btn],
        )
        copy_btn.click(
            fn=None,
            inputs=[output],
            outputs=None,
            # 纯前端复制：剪贴板只在 localhost（安全上下文）可用，本项目满足；
            # 成功后切 .copied 类（青蓝字/边框 + 淡青底）并变文案“已复制”，
            # 1.5 秒后恢复（elem_id 即按钮本身，兜底再查一层 button，
            # 防 Gradio 把 id 放在包装层上）
            js="""(text) => {
                if (!text) return;
                navigator.clipboard.writeText(text).then(() => {
                    const btn = document.querySelector('#copy-btn button') ?? document.querySelector('#copy-btn');
                    btn.textContent = '已复制';
                    btn.classList.add('copied');
                    setTimeout(() => { btn.textContent = '复制全文'; btn.classList.remove('copied'); }, 1500);
                });
            }""",
        )
        btn.click(run, inputs=url, outputs=[status, output, file, history_ds, history_state, btn, copy_btn])
        # 输入框里回车等同点按钮
        url.submit(run, inputs=url, outputs=[status, output, file, history_ds, history_state, btn, copy_btn])

        # 页面每次加载都刷新目录，保证刷新浏览器也能看到磁盘上的最新历史；
        # 另挂一个纯前端监听器把光标放进链接框，粘完链接直接回车
        demo.load(refresh_history, inputs=None, outputs=[history_ds, history_state])
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { const el = document.querySelector('#url-input input, #url-input textarea'); if (el) el.focus(); }",
        )
    return demo


if __name__ == "__main__":
    TMP_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    ensure_fonts()  # 界面字体就绪（缺失才下载，已有则跳过）
    load_recognizer()  # 启动即加载，首次转写不用再等
    # demo 暴露到模块级：`gradio app.py` 开发模式的热重载靠它查找并热替换界面
    demo = build_demo()
    demo.queue(max_size=8).launch(inbrowser=True, show_error=True)
