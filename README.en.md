# extract-pgs

[简体中文](README.md) · English

Convert PGS image subtitles from MKV, TS, M2TS, or standalone SUP files to sidecar SRT files. The queue processes one movie at a time. FFmpeg extracts the original tracks, Subtitle Edit's SeConv exports images and timecodes in batches that respect subtitle boundaries, and PaddleOCR recognizes text on the local CPU. The scripts validate results against the original timing, record per-movie state, and save approved SRT files alongside the video.

## Installation

Verified on Linux x86-64 with Python 3.13, FFmpeg 7.1, and SeConv 5.3.0. Make sure `ffmpeg` and `ffprobe` are on your PATH. Extract the SeConv package from [Subtitle Edit v5.3.0](https://github.com/SubtitleEdit/subtitleedit/releases/tag/v5.3.0) and add `seconv` to your PATH, or set `PGS_SECONV` to the full executable path.

```bash
git clone https://github.com/zhongkechen/extract-pgs.git
cd extract-pgs
export TMPDIR="$HOME/.cache/extract-pgs"
mkdir -p "$TMPDIR"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the [detection model](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) and the [recognition model](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx). Each local model directory must contain `inference.onnx`, `inference.yml`, and `inference.json`.

```bash
python -m pip install 'huggingface-hub==2.3.0'
export PGS_MODEL_ROOT="$HOME/.local/share/extract-pgs/models"
for model in PP-OCRv6_medium_det_onnx PP-OCRv6_medium_rec_onnx; do
  hf download "PaddlePaddle/$model" --include "inference.*" --local-dir "$PGS_MODEL_ROOT/$model"
done
```

This configuration uses the ONNX Runtime CPU backend. Once the dependencies and models are installed, OCR runs locally. See the [PaddleOCR inference engine documentation](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/local_inference/inference_engine.html).

## Prepare the conversion queue

This example uses `/path/to/movies/demo/Sample/Sample.mkv`. Set `PGS_MEDIA_ROOT` to your actual media root. Paths may contain spaces.

```bash
export PGS_MEDIA_ROOT=/path/to/movies
export PGS_WORK_DIR="$TMPDIR/queue"
mkdir -p "$PGS_WORK_DIR"

ffprobe -v error -select_streams s \
  -show_entries 'stream=index,codec_name:stream_tags=language,title' \
  -of json "$PGS_MEDIA_ROOT/demo/Sample/Sample.mkv"
```

Save the following as `$PGS_WORK_DIR/queue.md`. The queue parser expects the exact Chinese field labels and indentation shown below. Replace the example directory and filename with your own.

```markdown
# PGS 字幕转换

- 当前进度：**0 / 1**

- [ ] **Sample** | 状态：待处理 | 完成日期：— | 备注：—
  目录：`demo/Sample`
  来源：`demo/Sample/Sample.mkv`
```

Each movie directory is a two-level path relative to the media root, such as `demo/Sample`. For a movie with multiple sources, add a separate `来源：` line for each source. All source paths remain relative to the media root. The script reads every `hdmv_pgs_subtitle` track in each source. A standalone `.sup` requires a matching video in the same directory to determine duration and output filenames. If multiple videos are present, the SUP filename stem must match the corresponding video stem or start with that stem followed by a dot.

## Run and resume

Start with a movie whose subtitle language is known to generate candidate SRT files for review.

```bash
python convert_queue.py --only demo/Sample --prepare-only --offline
```

Candidates are named `track-*-final.srt` and appear under `$PGS_WORK_DIR/movie-*/`. The movie's `state.json` records source tracks, candidates, checksums, and intended output paths. Offline mode needs a clear language hint from the track title, language tag, or SUP filename. Tracks with ambiguous language remain pending.

The default review uses an authenticated `codex` CLI to send sampled subtitle images and the corresponding OCR text to OpenAI. Tracks without language metadata may also have a few images sent to select an OCR model. Built-in review requires Noto Sans CJK. Set `PGS_REVIEW_FONT` to override the font path. Set `PGS_REVIEW_MODEL` to a model available to your account. The default is `gpt-6.2-sol`.

```bash
# Review prepared candidates against the images and save approved sidecar SRT files.
python convert_queue.py --only demo/Sample --finalize-only

# Resume the queue in the same work directory and clean up reproducible intermediates after approval.
python convert_queue.py --resume --cleanup
```

Without `--only`, the queue follows manifest order. Use `--limit N` to cap the number of movies in a run and repeat `--prioritize MOVIE_KEY` to set processing priorities. After an interruption, run with `--resume` using the same media root, manifest, work directory, and model configuration. If source files, models, preprocessing, or recognition code change, the script checks existing evidence and rejects stale caches. The queue stops when the temporary drive has less than 2 GiB of free space.

After approval, sidecar SRT files are saved beside the matching video, for example `Sample.English.stream2.OCR.en.srt`. The script refuses to overwrite an unrecognized subtitle file with the same name. If an error occurs or any track still needs approval, the movie stays in the queue. Check its `state.json` and track logs for details. In prepare-only mode, reaching the pending-review state means preparation succeeded.

## Optional configuration

- `PGS_MANIFEST` sets the manifest path. The default is `$PGS_WORK_DIR/queue.md`.
- `PGS_WORK_DIR` sets the state and evidence directory. The default is `$TMPDIR/queue`. Keep it inside `TMPDIR`, with both paths outside `/tmp`.
- `PGS_MODEL_ROOT` sets the local ONNX model directory. The default is `~/.local/share/extract-pgs/models`.
- The default recognizer uses PP-OCRv6. Configure additional languages in `$PGS_WORK_DIR/extra-models.json`. This example uses a Russian PP-OCRv5 model. Replace the directory with your actual model path.

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

Hebrew can use a local Tesseract installation. Install `tesseract` and its Hebrew and English language data. For `he` in `extra-models.json`, set `engine: "tesseract"`, `language: "heb"`, `structure_validated: true`, `tessdata_dir`, and `model_file`. Set the last two fields to the directory containing `heb.traineddata` and the full path to that file, respectively. If the English data is outside the default Debian location, set `PGS_TESSDATA_ENG` to the path to `eng.traineddata`.

### Custom review

Use `--review-command 'python /path/to/reviewer.py'` to replace the final image and text review. The queue passes the path to `state.json` as the last argument. The review program should write `review.json` in the same directory with `movie_key`, `approved`, and `tracks`. Each nonduplicate track needs an entry with `track_id`, `approved`, `raw_sha256`, and `candidate_sha256`. The checksums must match the artifacts actually reviewed. To keep the entire process offline, prepare candidates with `--prepare-only --offline`, then run `--finalize-only` with a local custom reviewer.

With a custom reviewer, omit `--cleanup` to keep all intermediates. Cleanup requires evidence of the samples reviewed. In the track entry in `review.json`, record a list of 1-based `requested_indices`. In the corresponding track object in `state.json`, set `visual_review_directory`. Under its `original-images/` subdirectory, keep copies of the corresponding originals with names such as `000001.png`. The cache is retained if review evidence is missing.

This repository contains the reusable conversion queue, OCR adapters, timecode validation, and review logic. Media library manifests, review records, model weights, and source-specific repair rules live in their respective work environments.
