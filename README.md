# extract-pgs

简体中文 · [English](README.en.md)

把 MKV、TS / M2TS 或独立 SUP 中的 PGS 图片字幕转为外挂 SRT。队列按影片顺序运行。FFmpeg 提取原始轨道，Subtitle Edit 的 SeConv 按完整字幕边界分批导出图像和时间码，PaddleOCR 在本机 CPU 上识别文字。脚本用原始时间基线校验结果，为每部影片保留状态文件，并在校对通过后把独立字幕版本保存到视频旁边。

## 安装

在 Linux x86-64、Python 3.13、FFmpeg 7.1 和 SeConv 5.3.0 上验证。确认 `ffmpeg` 和 `ffprobe` 在 PATH 中。从 [Subtitle Edit v5.3.0](https://github.com/SubtitleEdit/subtitleedit/releases/tag/v5.3.0) 解压 SeConv 发行包，把其中的 `seconv` 加入 PATH，或把 `PGS_SECONV` 设为可执行文件的完整路径。

```bash
git clone https://github.com/zhongkechen/extract-pgs.git
cd extract-pgs
export TMPDIR="$HOME/.cache/extract-pgs"
mkdir -p "$TMPDIR"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

下载 [检测模型](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx)和[识别模型](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx)。每个模型的本地目录需保留 `inference.onnx`、`inference.yml` 和 `inference.json`。

```bash
python -m pip install 'huggingface-hub==2.3.0'
export PGS_MODEL_ROOT="$HOME/.local/share/extract-pgs/models"
for model in PP-OCRv6_medium_det_onnx PP-OCRv6_medium_rec_onnx; do
  hf download "PaddlePaddle/$model" --include "inference.*" --local-dir "$PGS_MODEL_ROOT/$model"
done
```

这套配置使用 ONNX Runtime 的 CPU 后端。安装好依赖和模型后，OCR 在本机运行。 [PaddleOCR 推理引擎文档](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/local_inference/inference_engine.html)

## 准备待转换清单

以下示例假设电影位于 `/path/to/movies/demo/Sample/Sample.mkv`。将 `PGS_MEDIA_ROOT` 改为实际媒体根目录。各路径可含空格。

```bash
export PGS_MEDIA_ROOT=/path/to/movies
export PGS_WORK_DIR="$TMPDIR/queue"
mkdir -p "$PGS_WORK_DIR"

ffprobe -v error -select_streams s \
  -show_entries 'stream=index,codec_name:stream_tags=language,title' \
  -of json "$PGS_MEDIA_ROOT/demo/Sample/Sample.mkv"
```

把下面的内容保存为 `$PGS_WORK_DIR/queue.md`。保留固定的中文字段名和缩进，把目录和文件名换成实际路径。

```markdown
# PGS 字幕转换

- 当前进度：**0 / 1**

- [ ] **Sample** | 状态：待处理 | 完成日期：— | 备注：—
  目录：`demo/Sample`
  来源：`demo/Sample/Sample.mkv`
```

每项电影目录为相对媒体根目录的两级路径，例如 `demo/Sample`。一部电影有多个来源时，逐行添加 `来源：`，路径仍从媒体根目录算起。脚本读取每个来源中的全部 `hdmv_pgs_subtitle` 轨。独立 `.sup` 需有同目录的匹配视频来确定时长和输出文件名。多个候选视频时，SUP 的文件名应与对应视频同名，或以视频基名加点号开头。

## 运行与恢复

先转换一部已标注语言的影片，生成可检查的候选 SRT。

```bash
python convert_queue.py --only demo/Sample --prepare-only --offline
```

候选文件位于 `$PGS_WORK_DIR/movie-*/` 下，名为 `track-*-final.srt`。同一部影片的 `state.json` 记录源轨、候选文件、校验值和预计发布路径。离线模式需要轨道标题、语言标签或 SUP 文件名能提供明确的语言线索，无法判断时会留待处理。

默认校对流程用已登录的 `codex` CLI 调用 OpenAI，发送待核对的字幕图像与对应 OCR 文本。未标注语言的轨道也可能发送少量字幕图像以选择 OCR 模型。使用内置校对时需装有 Noto Sans CJK，可通过 `PGS_REVIEW_FONT` 指定字体路径。可用 `PGS_REVIEW_MODEL` 指定账号可用的模型，默认 `gpt-6.2-sol`。

```bash
# 对已准备的候选做图文校对，通过后保存外挂 SRT。
python convert_queue.py --only demo/Sample --finalize-only

# 从同一工作目录恢复整个队列，通过校对后清理可再生成的中间文件。
python convert_queue.py --resume --cleanup
```

不指定 `--only` 时按照清单顺序处理。用 `--limit N` 限制本次影片数，用可重复的 `--prioritize 目录键` 指定优先顺序。中断后使用同一媒体根目录、清单、工作目录和模型配置执行 `--resume`。如修改了源文件、模型、预处理或识别代码，脚本会检查已有证据，拒绝不匹配的缓存。临时盘可用空间低于 2 GiB 时，队列会停止。

校对通过后，外挂 SRT 保存到对应视频旁边，例如 `Sample.English.stream2.OCR.en.srt`。脚本拒绝覆盖未知的同名字幕。出现异常或有轨道尚未获批时，影片继续留在清单，查看对应的 `state.json` 和轨道日志。仅准备候选时，进入待校对状态即表示本阶段成功。

## 可选配置

- `PGS_MANIFEST` 指定清单路径，默认是 `$PGS_WORK_DIR/queue.md`。
- `PGS_WORK_DIR` 指定状态和证据目录，默认为 `$TMPDIR/queue`。工作目录须位于 `TMPDIR` 下，两者都应避开 `/tmp`。
- `PGS_MODEL_ROOT` 指定本地 ONNX 模型目录，默认是 `~/.local/share/extract-pgs/models`。
- 默认识别器使用 PP-OCRv6。其他语言可在 `$PGS_WORK_DIR/extra-models.json` 显式配置。下面是俄语 PP-OCRv5 的示例，把路径换成实际模型目录。

```json
{
  "ru": {
    "engine": "paddle",
    "paddle_language": "ru",
    "model_name": "eslav_PP-OCRv5_mobile_rec",
    "model_dir": "/path/to/eslav_PP-OCRv5_mobile_rec_onnx",
    "structure_validated": true
  }
}
```

希伯来文可用可选的本地 Tesseract。安装 `tesseract` 及希伯来文和英文数据，在 `extra-models.json` 中为 `he` 设置 `engine: "tesseract"`、`language: "heb"`、`structure_validated: true`、`tessdata_dir` 和 `model_file`。后两项分别为 `heb.traineddata` 所在目录和文件的完整路径。若英文数据不在 Debian 默认位置，通过 `PGS_TESSDATA_ENG` 指定 `eng.traineddata`。

### 自定义校对

使用 `--review-command 'python /path/to/reviewer.py'` 可替换最后的图文校对。队列会将 `state.json` 的路径作为最后一个参数传入。校对程序应在同目录写入 `review.json`，包含 `movie_key`、`approved` 和 `tracks`。每条非重复轨须有一项，包含 `track_id`、`approved`、`raw_sha256` 和 `candidate_sha256`。这些校验值须与实际复核的文本一致。先用 `--prepare-only --offline` 准备候选，再用自定义校对执行 `--finalize-only`，可让整个流程保持离线。

自定义校对时可省略 `--cleanup`，保留全部中间文件。使用 `--cleanup` 还需要提供抽查证据。在 `review.json` 的轨道条目中记录一组从 1 开始的 `requested_indices`，在 `state.json` 的对应轨道对象中记录 `visual_review_directory`，并在该目录的 `original-images/` 下保留对应的原图副本，文件名格式为 `000001.png`。缺少核验依据时会保留缓存。

本仓库包含可复用的转换队列、OCR 适配器、时间码验证和校对逻辑。原媒体库清单、校对记录、模型权重及按具体片源编写的修复规则保存在各自的工作环境中。
