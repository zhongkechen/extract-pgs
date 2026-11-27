#!/usr/bin/env python3
"""Sequential PGS conversion queue. Importing this file never starts work.

Each movie is extracted, recognized, reviewed and published before advancing.
Unresolved movies remain unchecked. All intermediates live under this run's
TMPDIR; existing media and subtitles are never overwritten.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import uuid

from pgs_config import CODE_DIR, MANIFEST, MODEL_ROOT, SECONV, TESSDATA_ENG, TMP_ROOT, WORK
from progress import atomic_write_json, batch_lock, parse_movies, update_movie
from validation import (compare_tracks, merge_adjacent, parse_srt, publishable_entries, read_bdn,
                        validate_ocr_srt, write_srt)

PYTHON = sys.executable
VIDEO_EXTS = {'.mkv', '.mp4', '.m2ts', '.mts', '.ts', '.avi', '.rmvb', '.rm',
              '.flv', '.mov', '.m4v', '.wmv', '.webm', '.mpg', '.mpeg', '.vob',
              '.ogm', '.ogv', '.asf', '.3gp', '.mxf', '.divx', '.m2v', '.f4v'}
ISO = {
    'chi': 'ch', 'zho': 'ch', 'zh': 'ch', 'eng': 'en', 'en': 'en',
    'jpn': 'japan', 'ja': 'japan', 'japan': 'japan',
    'fre': 'fr', 'fra': 'fr', 'ger': 'de', 'deu': 'de', 'spa': 'es',
    'por': 'pt', 'ita': 'it', 'dut': 'nl', 'nld': 'nl', 'ind': 'id',
    'cze': 'cs', 'ces': 'cs', 'hun': 'hu', 'tur': 'tr', 'est': 'et',
    'pol': 'pl', 'lit': 'lt', 'fin': 'fi', 'rum': 'ro', 'ron': 'ro',
    'slv': 'sl', 'swe': 'sv', 'lav': 'lv', 'hrv': 'hr', 'dan': 'da',
    'ice': 'is', 'isl': 'is', 'nor': 'no', 'nob': 'no', 'may': 'ms',
    'msa': 'ms', 'slo': 'sk', 'slk': 'sk', 'vie': 'vi', 'cat': 'ca',
    'kor': 'korean', 'ko': 'korean', 'tha': 'th', 'rus': 'ru', 'ukr': 'uk',
    'bul': 'bg', 'ara': 'ar', 'hin': 'hi', 'heb': 'he', 'gre': 'el',
    'ell': 'el', 'srp': 'rs_cyrillic', 'scc': 'rs_cyrillic',
}
JELLYFIN = {'ch': 'zh', 'chinese_cht': 'zh', 'japan': 'ja', 'korean': 'ko',
            'rs_cyrillic': 'sr', 'rs_latin': 'sr', 'heb': 'he'}
V6_LANGS = set('ch chinese_cht en japan af az bs cs cy da de es et fr ga hr hu id is it ku la lt lv mi ms mt nl no oc pl pt ro rs_latin sk sl sq sv sw tl tr uz vi french german fi eu gl lb rm ca qu'.split())


class BatchStop(RuntimeError):
    """A condition that must stop the entire queue, rather than skip a movie."""


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def log(message):
    print(f'[{now()}] {message}', flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(path):
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def check_space():
    free = shutil.disk_usage(WORK).free
    if free < 2 * 1024 ** 3:
        raise BatchStop(f'TMPDIR 剩余空间不足 2 GiB（{free / 1024**3:.2f} GiB），保留状态后停止')


def save(state):
    state['updated_at'] = now()
    atomic_write_json(Path(state['work_dir']) / 'state.json', state)


def env_for(track=None):
    env = os.environ.copy()
    runtime = WORK / 'runtime'
    for name in ('paddlex', 'cache', 'config', 'data', 'dotnet', 'huggingface'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    env.update(PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PADDLE_PDX_CACHE_HOME=str(runtime / 'paddlex'),
               XDG_CACHE_HOME=str(runtime / 'cache'), XDG_CONFIG_HOME=str(runtime / 'config'),
               XDG_DATA_HOME=str(runtime / 'data'), HF_HOME=str(runtime / 'huggingface'),
               DOTNET_BUNDLE_EXTRACT_BASE_DIR=str(runtime / 'dotnet'))
    env['TMPDIR'] = str(TMP_ROOT)
    if track:
        env['PGS_TRACK_WORK'] = track['evidence_dir']
        for key in ('PGS_REC_MODEL_NAME', 'PGS_REC_MODEL_DIR', 'PGS_REC_DICT_SHA256'):
            env.pop(key, None)
        model = track.get('model', {})
        for source, target in [('model_name', 'PGS_REC_MODEL_NAME'),
                               ('model_dir', 'PGS_REC_MODEL_DIR'),
                               ('dict_sha256', 'PGS_REC_DICT_SHA256')]:
            if model.get(source):
                env[target] = str(model[source])
        if model.get('tessdata_dir'):
            env['PGS_TESSDATA_DIR'] = str(model['tessdata_dir'])
            env['TESSDATA_PREFIX'] = str(model['tessdata_dir'])
    return env


def run_command(command, logfile, *, env=None, progress=None, stderr_path=None):
    """Wait synchronously; reap the child before ever advancing to another movie."""
    check_space()
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w', encoding='utf-8') as out:
        err = Path(stderr_path).open('w', encoding='utf-8') if stderr_path else None
        try:
            proc = subprocess.Popen([str(arg) for arg in command], stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err or subprocess.STDOUT, env=env or env_for(),
                                    start_new_session=True)
        except BaseException:
            if err:
                err.close()
            raise
        last = 0
        try:
            while proc.poll() is None:
                check_space()
                if progress and time.monotonic() - last >= 60:
                    progress()
                    last = time.monotonic()
                time.sleep(2)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
        finally:
            if err:
                err.close()
    if proc.returncode:
        tail = logfile.read_text(errors='replace')[-1800:]
        raise RuntimeError(f'命令退出 {proc.returncode}；日志 {logfile}：{tail}')


def probe(path):
    before = snapshot(path)
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'format=duration,start_time:stream=index,codec_name,codec_type:stream_tags=language,title',
                        '-of', 'json', str(path)], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=120, check=True)
    data = json.loads(p.stdout)
    if snapshot(path) != before:
        raise RuntimeError(f'探测期间源文件变化：{path}')
    data['fingerprint'] = before
    data['diagnostics'] = p.stderr.strip()
    return data


def companion_video(sup):
    candidates = sorted(p for p in sup.parent.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    named = [p for p in candidates if sup.stem == p.stem or sup.stem.startswith(p.stem + '.')]
    if len(named) == 1:
        return named[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f'独立 SUP 对应视频有歧义，需确认：{sup}')


def infer_language(track):
    """Use explicit version labels first; never turn an unknown tag into Chinese."""
    title = track.get('title', '')
    hints = title + ' ' + (Path(track['source']).stem if track['source'].lower().endswith('.sup') else '')
    lower = hints.casefold()
    if re.search(r'繁[体體中]|[简簡]繁|cht|zh[-_.]?(tw|hant)', lower):
        return 'chinese_cht'
    if re.search(r'[简簡][体體中]|chs|zh[-_.]?(cn|hans)|中英|中文', lower):
        return 'ch'
    if re.search(r'(^|[^a-z])(eng|english)([^a-z]|$)|英文|英语|英語', lower):
        return 'en'
    if re.search(r'日[文语語]|japanese', lower):
        return 'japan'
    raw = str(track.get('language') or 'und').lower()
    return ISO.get(raw, raw) if raw not in {'und', 'unknown', ''} else None


def choose_model(language):
    language = ISO.get(language, language)
    if language in V6_LANGS:
        return language, {'engine': 'paddle', 'version': 'PP-OCRv6'}
    path = WORK / 'extra-models.json'
    extra = json.loads(path.read_text()) if path.exists() else {}
    model = extra.get(language)
    if not isinstance(model, dict) or not model.get('engine', 'paddle') in {'paddle', 'tesseract'}:
        raise RuntimeError(f'语言 {language!r} 的经过验证模型尚不可用')
    if model.get('engine') != 'tesseract' and not all(model.get(k) for k in ('model_name', 'model_dir')):
        raise RuntimeError(f'语言模型配置不完整：{language}')
    return model.get('paddle_language') or model.get('language') or language, model


def safe_label(value):
    value = re.sub(r'[\x00-\x1f/\\:*?"<>|]', '_', str(value)).strip(' .')
    result = ''
    for ch in value or '字幕':
        if len((result + ch).encode('utf-8')) > 80:
            break
        result += ch
    return result


def output_path(track):
    video = Path(track['video_path'])
    label = safe_label(track.get('title') or track.get('language') or track['paddle_language'])
    lang = JELLYFIN.get(track['paddle_language'], track['paddle_language'])
    if track.get('language_identification'):
        # Script identification chooses a model, not necessarily an exact
        # subtitle language (the v6 model covers Chinese/Japanese/Latin text).
        lang = track['language_identification'].get('subtitle_language') or 'und'
    sidecar = '.sup' if track['source'].lower().endswith('.sup') else ''
    suffix = f'.{label}{sidecar}.stream{track["index"]}.OCR.{lang}.srt'
    candidate = video.with_name(video.stem + suffix)
    if len(candidate.name.encode('utf-8')) > 255:
        suffix = f'.{sidecar.strip(".") + "." if sidecar else ""}stream{track["index"]}.OCR.{lang}.srt'
        candidate = video.with_name(video.stem + suffix)
    if len(candidate.name.encode('utf-8')) > 255:
        raise RuntimeError(f'视频文件名过长，无法安全追加外置字幕名：{video.name}')
    return candidate


def work_for(movie):
    return WORK / ('movie-' + hashlib.sha256(movie['key'].encode()).hexdigest()[:16])


def state_for(movie, resume):
    directory = work_for(movie)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'state.json'
    if path.exists():
        if not resume:
            raise RuntimeError(f'已有状态；请使用 --resume：{path}')
        state = json.loads(path.read_text())
        if state['key'] != movie['key']:
            raise RuntimeError('工作目录中的电影标识不匹配')
        return state
    return {'schema': 1, 'key': movie['key'], 'title': movie['title'],
            'work_dir': str(directory), 'sources': {}, 'tracks': {},
            'errors': [], 'status': 'new', 'created_at': now()}


def prepare_sources(movie, state, media_root):
    directory = Path(state['work_dir'])
    state['source_scan_complete'] = False
    expected_ids = []
    save(state)
    for relative in movie['sources']:
        movie_dir = (media_root / movie['key']).resolve(strict=True)
        source = (media_root / relative).resolve(strict=True)
        if not movie_dir.is_relative_to(media_root) or not source.is_relative_to(movie_dir) or not source.is_file():
            raise ValueError(f'字幕源须为电影目录内的文件：{relative}')
        metadata = probe(source)
        streams = [s for s in metadata.get('streams', []) if s.get('codec_name') == 'hdmv_pgs_subtitle']
        if not streams:
            raise RuntimeError(f'此前PGS来源已不含可识别PGS轨道，需复查：{source}')
        video = companion_video(source) if source.suffix.lower() == '.sup' else source
        video = video.resolve(strict=True)
        if not video.is_relative_to(movie_dir):
            raise ValueError(f'SUP 对应视频须位于电影目录内：{video}')
        video_meta = probe(video) if video != source else metadata
        duration = float(video_meta.get('format', {}).get('duration', 0))
        if duration <= 0:
            raise RuntimeError(f'无法确认正片时长：{video}')
        previous = state['sources'].get(str(source))
        if previous and previous['fingerprint'] != metadata['fingerprint']:
            raise RuntimeError(f'源文件自本次任务开始后改变，拒绝沿用旧字幕：{source}')
        if previous and (previous.get('video_path') != str(video)
                         or previous.get('video_fingerprint') != video_meta['fingerprint']):
            raise RuntimeError(f'SUP对应正片或其版本发生变化，拒绝沿用旧状态：{video}')
        start_time = float(metadata.get('format', {}).get('start_time') or 0)
        normalize = source.suffix.lower() in {'.ts', '.m2ts', '.mts'}
        timeline = {'source_start_time_seconds': start_time,
                    'offset_seconds': -start_time if normalize else 0.0,
                    'method': 'ffmpeg-copyts-start_at_zero' if normalize else 'preserve-original'}
        if source.suffix.lower() == '.sup':
            timeline = {'source_start_time_seconds': start_time, 'offset_seconds': 0.0,
                        'method': 'preserve-independent-sup'}
        if previous and previous.get('extracted') and normalize and previous.get('timeline') != timeline:
            raise RuntimeError(f'旧TS/M2TS提取未绑定正确起始偏移，不能复用：{source}')
        source_id = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        base = directory / source_id
        base.mkdir(parents=True, exist_ok=True)
        entry = {'fingerprint': metadata['fingerprint'], 'duration': duration,
                 'video_path': str(video), 'video_fingerprint': video_meta['fingerprint'],
                 'metadata': metadata, 'work_dir': str(base), 'timeline': timeline}
        if previous:
            entry.update({k: v for k, v in previous.items() if k.startswith('extract')})
        state['sources'][str(source)] = entry
        paths = {}
        for stream in streams:
            index = stream['index']
            track_id = f'{source_id}:stream{index}'
            expected_ids.append(track_id)
            tags = stream.get('tags') or {}
            track = state['tracks'].setdefault(track_id, {})
            track.update(id=track_id, source=str(source), index=index, video_path=str(video),
                         duration=duration, language=tags.get('language', 'und'),
                         title=tags.get('title', ''),
                         sup=str(base / f'track-{index}.sup'),
                         baseline=str(base / f'track-{index}-timing' / f'track-{index}.srt'),
                         bdn=str(base / f'track-{index}-images' / f'track-{index}' / 'index.xml'),
                         raw_srt=str(base / f'track-{index}-ocr' / f'track-{index}.srt'),
                         candidate_srt=str(base / f'track-{index}-final.srt'),
                         ocr_json_dir=str(base / f'track-{index}-ocr' / 'json'),
                         evidence_dir=str(base / f'track-{index}-work'),
                         flags_path=str(base / f'track-{index}-work' / 'paddle-flags.json'))
            track.setdefault('status', 'new')
            paths[track_id] = Path(track['sup'])
        # Adopt only the explicitly prepared first-movie extraction and snapshot.
        if entry.get('extracted'):
            for tid, path in paths.items():
                if not path.exists() or digest(path) != entry.get('extracted_hashes', {}).get(tid):
                    raise RuntimeError(f'已提取 SUP 缺失或校验值改变：{path}')
        else:
            # A prior interrupted command owns only these pre-recorded temporary files.
            for tid, path in paths.items():
                if path.exists():
                    if str(path) not in entry.get('extract_owned_paths', []):
                        raise RuntimeError(f'拒绝覆盖未知临时 SUP：{path}')
                    path.unlink()
            entry['extract_owned_paths'] = [str(p) for p in paths.values()]
            save(state)
            update_movie(state['key'], '转换中', f'正在提取 {source.name} 的 {len(streams)} 条 PGS')
            if source.suffix.lower() == '.sup':
                with source.open('rb') as src, next(iter(paths.values())).open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            else:
                command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
                           '-copyts']
                if normalize:
                    command.append('-start_at_zero')
                command += ['-i', str(source)]
                for tid, path in paths.items():
                    command += ['-map', f'0:{state["tracks"][tid]["index"]}', '-c:s', 'copy', str(path)]
                run_command(command, base / 'extract.log')
            entry['extracted_hashes'] = {tid: digest(p) for tid, p in paths.items()}
            entry['extracted'] = True
        for tid, path in paths.items():
            state['tracks'][tid]['sup_sha256'] = entry['extracted_hashes'][tid]
        assert_unchanged(state)
        save(state)
    if set(expected_ids) != set(state['tracks']):
        raise RuntimeError('当前PGS轨道集合与本批次状态不一致，需检查旧状态')
    state['expected_track_ids'] = expected_ids
    state['source_scan_complete'] = True
    save(state)


def assert_unchanged(state):
    for path, source in state['sources'].items():
        if snapshot(path) != source['fingerprint']:
            raise RuntimeError(f'源文件发生变化：{path}')
        if snapshot(source['video_path']) != source['video_fingerprint']:
            raise RuntimeError(f'SUP对应正片发生变化：{source["video_path"]}')


def seconv_output(track, mode):
    expected = Path(track['baseline'] if mode == 'subrip' else track['bdn'])
    if expected.exists():
        return
    output_dir = expected.parent if mode == 'subrip' else expected.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == 'bdnxml':
        from bounded_bdn import export_bdn
        log(f'stream {track["index"]}: 按完整字幕边界分批导出图片')
        export_bdn(Path(track['sup']), output_dir, expected, run_command, Path(track['baseline']))
        return
    command = [SECONV, track['sup'], mode, f'--output-folder:{output_dir}', '--json']
    if mode == 'subrip':
        command.append('--time-codes-only')
    logfile = output_dir / 'seconv-result.json'
    run_command(command, logfile, stderr_path=output_dir / 'seconv.stderr.log')
    result = json.loads(logfile.read_text(encoding='utf-8-sig'))
    if not result.get('success') or result.get('failedFiles') or not expected.exists():
        raise RuntimeError(f'seconv 输出不完整：{logfile}')


def rgba_signature(events):
    from PIL import Image
    h = hashlib.sha256()
    for event in events:
        with Image.open(event['image_path']) as image:
            rgba = image.convert('RGBA')
            h.update(json.dumps([event['start_ms'], event['end_ms'], *rgba.size]).encode() + b'\0')
            h.update(rgba.tobytes())
    return h.hexdigest()


def read_ocr_entries(track, events):
    entries = []
    for event in events:
        result_file = Path(track['ocr_json_dir']) / (Path(event['image_path']).stem + '_res.json')
        data = json.loads(result_file.read_text())
        data = data.get('res', data)
        texts = data.get('rec_texts')
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise RuntimeError(f'OCR JSON 缺少 rec_texts：{result_file}')
        entries.append({'index': event['index'], 'start_ms': event['start_ms'],
                        'end_ms': event['end_ms'], 'text': '\n'.join(texts)})
    return entries


def resolve_language(state, track, *, offline):
    language = infer_language(track)
    if not language:
        if offline:
            raise RuntimeError('字幕语言未标注，离线模式需要可识别的轨道标题或语言标签')
        module = importlib.import_module('visual_review')
        if not hasattr(module, 'identify_language'):
            raise RuntimeError('字幕语言未标注，图文语言确认接口尚不可用')
        save(state)
        events = read_bdn(track['bdn'], track['baseline'])
        indices = sorted({round((len(events) - 1) * fraction) for fraction in (0, .25, .5, .75, 1)})
        identified = module.identify_language(
            [events[index]['image_path'] for index in indices],
            Path(track['evidence_dir']) / 'language-identification',
            metadata={'source': track['source'], 'index': track['index'],
                      'language': track['language'], 'title': track['title']})
        if not isinstance(identified, dict) or identified.get('recognized') is not True or not identified.get('language'):
            raise RuntimeError('图文核对未能明确识别字幕语言')
        track['language_identification'] = identified
        language = identified.get('paddle_language') or ISO.get(identified['language'], identified['language'])
    track['paddle_language'], track['model'] = choose_model(language)
    track['output'] = str(output_path(track))


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _adapter_module():
    """Load only definitions, never the adapter's CLI or an OCR model."""
    path = CODE_DIR / 'paddle_batch.py'
    namespace = {'__name__': '_pgs_signature_adapter', '__file__': str(path)}
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def recognition_identity(track, events):
    """Bind resumable outputs to pixels/times, actual weights and OCR code."""
    model = track['model']
    engine = model.get('engine', 'paddle')
    adapter = CODE_DIR / ('ocr_tesseract.py' if engine == 'tesseract' else 'paddle_batch.py')
    algorithms = {adapter.name: digest(adapter)}
    if engine == 'tesseract':
        algorithms['paddle_batch.py'] = digest(CODE_DIR / 'paddle_batch.py')
        weights = [Path(model['tessdata_dir']) / 'heb.traineddata',
                   TESSDATA_ENG]
        configuration = {
            'engine': 'tesseract', 'language': 'heb+eng', 'psm': 6,
            'preprocess': 'RGBA-on-black-alpha-bbox-grayscale-invert-white-border24-v1',
            'version': subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0],
            'models': {path.name: digest(path) for path in weights}}
    else:
        import yaml
        module = _adapter_module()
        rec_dir = Path(model.get('model_dir') or MODEL_ROOT / 'PP-OCRv6_medium_rec_onnx').resolve(strict=True)
        rec_name = model.get('model_name') or 'PP-OCRv6_medium_rec'
        cfg = yaml.safe_load((rec_dir / 'inference.yml').read_text())
        if cfg.get('Global', {}).get('model_name') != rec_name:
            raise RuntimeError('识别模型名称与实际配置不一致')
        characters = cfg.get('PostProcess', {}).get('character_dict')
        if not isinstance(characters, list) or not characters:
            raise RuntimeError('识别模型缺少字符字典')
        dictionary_hash = hashlib.sha256(json.dumps(characters, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if model.get('dict_sha256') and model['dict_sha256'] != dictionary_hash:
            raise RuntimeError('识别模型字典已改变')
        configuration = {
            'settings': module.SETTINGS, 'language': track['paddle_language'],
            'versions': {p: importlib.metadata.version(p) for p in ('paddleocr', 'paddlex', 'onnxruntime', 'Pillow')},
            'models': {},
            'recognizer': {'model_name': rec_name, 'model_dir': str(rec_dir),
                           'ocr_version': 'PP-OCRv6' if rec_name == 'PP-OCRv6_medium_rec' else 'PP-OCRv5',
                           'character_count': len(characters), 'dictionary_sha256': dictionary_hash}}
        for role, directory in (('detector', MODEL_ROOT / 'PP-OCRv6_medium_det_onnx'), ('recognizer', rec_dir)):
            for path in sorted(directory.rglob('*')):
                relative = path.relative_to(directory)
                if path.is_file() and not any(part.startswith('.') for part in relative.parts):
                    configuration['models'][role + '/' + relative.as_posix()] = digest(path)
    identity = {'schema': 1, 'sup_sha256': digest(track['sup']),
                'decoded_signature': rgba_signature(events), 'language': track['paddle_language'],
                'engine': engine, 'adapter_algorithms': algorithms,
                'configuration': configuration,
                'srt_assembly': 'all-rec-texts-in-order-newlines-exact-baseline-v1'}
    if identity['sup_sha256'] != track.get('sup_sha256'):
        raise RuntimeError('源SUP校验值与提取记录不一致')
    return identity, _json_hash(identity)


def verify_completed_ocr(track, events, identity):
    """Read every recorded input and result; no inference or file writes."""
    from PIL import Image, ImageOps
    evidence_root = Path(track['evidence_dir']).resolve()
    progress = json.loads((evidence_root / 'paddle-progress.json').read_text())
    if (progress.get('status') != 'completed' or progress.get('language') != track['paddle_language']
            or progress.get('completed') != len(events) or progress.get('total') != len(events)):
        raise RuntimeError('OCR批次尚未完成或条目/语言不一致')
    evidence = Path(progress['evidence']).resolve(strict=True)
    if not evidence.is_relative_to(evidence_root / 'paddle-evidence'):
        raise RuntimeError('OCR证据路径未归属本轨道')
    configuration = json.loads((evidence / 'configuration.json').read_text())
    if _json_hash(configuration) != progress.get('signature'):
        raise RuntimeError('OCR配置与已记录signature不一致')
    expected = identity['configuration']
    if configuration != expected:
        raise RuntimeError('已有OCR使用的模型、参数、预处理或软件版本与当前计划不一致')
    rows = [json.loads(line) for line in (evidence / 'mapping.jsonl').read_text().splitlines()]
    expected_stems = [Path(event['image_path']).stem for event in events]
    if (len(rows) != len(events) or [r.get('index') for r in rows] != list(range(len(events)))
            or [r.get('basename') for r in rows] != sorted(expected_stems)):
        raise RuntimeError('OCR mapping没有逐项覆盖当前BDN图像')
    by_stem = {row['basename']: row for row in rows}
    if len(by_stem) != len(events):
        raise RuntimeError('OCR mapping中出现重复图像名称')
    output = Path(track['ocr_json_dir']).resolve()
    expected_files = {stem + '_res.json' for stem in expected_stems}
    if {p.name for p in output.iterdir() if p.is_file()} != expected_files:
        raise RuntimeError('OCR结果文件集合与当前输入不一致')
    module = _adapter_module() if identity['engine'] == 'paddle' else None
    result_hashes = {}
    for event in events:
        row = by_stem[Path(event['image_path']).stem]
        source = Path(event['image_path']).resolve(strict=True)
        original = Path(row['original']).resolve(strict=True)
        prepared_file = Path(row['prepared']).resolve(strict=True)
        result_file = Path(row['result']).resolve(strict=True)
        if (Path(row['source']).resolve() != source or not original.is_relative_to(evidence / 'original')
                or not prepared_file.is_relative_to(evidence / 'prepared') or not result_file.is_relative_to(evidence / 'results')):
            raise RuntimeError('OCR mapping跨越了当前输入或证据目录')
        if digest(source) != row.get('original_sha256') or digest(original) != row.get('original_sha256'):
            raise RuntimeError('已有OCR对应的原图并非当前BDN图像')
        with Image.open(source) as image:
            if module:
                prepared, transform = module.prepare_image(image)
                image_key = hashlib.sha256(progress['signature'].encode() + repr((prepared.mode, prepared.size)).encode() + prepared.tobytes()).hexdigest()
                if row.get('image_key') != image_key or row.get('transform') != transform:
                    raise RuntimeError('OCR预处理像素签名不匹配')
            else:
                rgba = image.convert('RGBA')
                bbox = rgba.getchannel('A').getbbox()
                if bbox is not None:
                    rgba = rgba.crop(bbox)
                background = Image.new('RGBA', rgba.size, (0, 0, 0, 255))
                background.alpha_composite(rgba)
                prepared = ImageOps.expand(ImageOps.invert(background.convert('L')), border=24, fill=255)
        with Image.open(prepared_file) as recorded:
            if (recorded.mode, recorded.size, recorded.tobytes()) != (prepared.mode, prepared.size, prepared.tobytes()):
                raise RuntimeError('保存的OCR输入与当前预处理结果不一致')
        destination = output / (row['basename'] + '_res.json')
        if digest(destination) != digest(result_file):
            raise RuntimeError('OCR输出与映射记录中的原始结果不一致')
        result_hashes[destination.name] = digest(destination)
    return {'configuration_signature': progress['signature'], 'evidence': str(evidence),
            'mapping_sha256': digest(evidence / 'mapping.jsonl'), 'result_hashes': result_hashes,
            'verified_inputs': len(events), 'verified_at': now()}


def _raw_is_owned(track, events):
    raw = Path(track['raw_srt'])
    if not raw.exists():
        return True
    if track.get('raw_sha256') and digest(raw) == track['raw_sha256']:
        return True
    try:
        return [(r['start_ms'], r['end_ms'], r['text']) for r in parse_srt(raw)] == [
            (r['start_ms'], r['end_ms'], r['text']) for r in read_ocr_entries(track, events)]
    except (OSError, ValueError, KeyError):
        return False


def quarantine_recognition(state, track, events):
    """Preserve superseded task outputs; never overwrite an untracked edit."""
    work = Path(state['work_dir']).resolve()
    raw, candidate = Path(track['raw_srt']), Path(track['candidate_srt'])
    if not _raw_is_owned(track, events):
        raise RuntimeError('原始SRT有未知手工编辑；保留文件，拒绝自动重识别覆盖')
    if candidate.exists():
        known = track.get('candidate_sha256') and digest(candidate) == track['candidate_sha256']
        if not known:
            raise RuntimeError('候选SRT缺少匹配的本任务校验值；保留文件')
    root = work / 'superseded-ocr' / (track['id'].replace(':', '_') + '-' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / 'previous-track-state.json', track)
    evidence = Path(track['evidence_dir'])
    paths = [raw, candidate, Path(track['ocr_json_dir']), evidence / 'paddle-evidence',
             evidence / 'paddle-cache', evidence / 'paddle-progress.json', evidence / 'paddle-flags.json']
    for path in paths:
        if path.exists():
            if not path.resolve().is_relative_to(work):
                raise RuntimeError('旧识别结果路径越出本电影工作目录')
            target = root / path.relative_to(work)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
    track.setdefault('recognition_history', []).append({'path': str(root), 'fingerprint': track.get('recognition_fingerprint')})
    for key in ('raw_sha256', 'candidate_sha256', 'recognition_proof', 'pre_review_validation',
                'empty_text_indices', 'visual_review_status', 'corrections',
                'recognition_adopted_from_verified_first_movie'):
        track.pop(key, None)


def bind_recognition(state, track, events, identity, fingerprint):
    previous = track.get('recognition_fingerprint')
    artifacts = any(Path(track[key]).exists() for key in ('raw_srt', 'candidate_srt'))
    json_dir = Path(track['ocr_json_dir'])
    artifacts = artifacts or (json_dir.exists() and any(json_dir.iterdir()))
    evidence = Path(track['evidence_dir'])
    artifacts = artifacts or (evidence / 'paddle-progress.json').exists()
    artifacts = artifacts or any((evidence / name).is_dir() and any((evidence / name).iterdir())
                                 for name in ('paddle-cache', 'paddle-evidence'))
    if previous == fingerprint:
        if Path(track['raw_srt']).exists() and not _raw_is_owned(track, events):
            raise RuntimeError('识别指纹一致但SRT出现未知编辑；拒绝复用或覆盖')
    elif artifacts:
        quarantine_recognition(state, track, events)
    track['recognition_configuration'] = identity
    track['recognition_fingerprint'] = fingerprint
    save(state)


def nontext_options(track):
    return {'reviewed_nontext': track.get('reviewed_nontext'),
            'source_sup': track['sup'], 'bdn': track['bdn']}


def candidate_entries(track, raw_entries):
    """Keep all raw events; omit only revalidated nontext events from output."""
    return merge_adjacent(publishable_entries(
        raw_entries, baseline=track['baseline'], **nontext_options(track)))


def prepare_track(state, track, representatives, *, offline):
    check_space()
    seconv_output(track, 'subrip')
    seconv_output(track, 'bdnxml')
    events = read_bdn(track['bdn'], track['baseline'])
    track['entries'] = len(events)
    signature = rgba_signature(events)
    track['decoded_signature'] = signature
    if signature in representatives:
        other = state['tracks'][representatives[signature]]
        comparison = compare_tracks(events, read_bdn(other['bdn'], other['baseline']))
        if comparison['identical']:
            track.update(status='duplicate', duplicate_of=other['id'], duplicate_comparison=comparison)
            save(state)
            log(f'{state["title"]} / stream {track["index"]}: 与 {other["id"]} 全部时间和像素一致，复用')
            return
    track.pop('duplicate_of', None)
    representatives[signature] = track['id']
    resolve_language(state, track, offline=offline)
    identity, fingerprint = recognition_identity(track, events)
    bind_recognition(state, track, events, identity, fingerprint)
    track['status'] = 'recognizing'
    save(state)
    json_dir = Path(track['ocr_json_dir'])
    json_dir.mkdir(parents=True, exist_ok=True)
    Path(track['evidence_dir']).mkdir(parents=True, exist_ok=True)
    # Reuse is allowed only after binding source/pixels, current model files,
    # software versions and the adapter algorithm to this run's state.
    raw = Path(track['raw_srt'])
    if not raw.exists():
        progress_file = Path(track['evidence_dir']) / 'paddle-progress.json'
        prior = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        complete_json = False
        if prior.get('status') == 'completed':
            try:
                track['recognition_proof'] = verify_completed_ocr(
                    track, events, identity,
                    allow_first_v6=bool(track.get('recognition_adopted_from_verified_first_movie')))
                complete_json = True
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                # Preserve the entire suspect result set before fresh inference.
                track['recognition_reuse_rejected'] = str(exc)
                quarantine_recognition(state, track, events)
                json_dir.mkdir(parents=True, exist_ok=True)
                save(state)

        def heartbeat():
            detail = ''
            if progress_file.exists():
                try:
                    p = json.loads(progress_file.read_text())
                    detail = f'，{p.get("completed", 0)}/{p.get("total", len(events))} 图'
                except (OSError, ValueError):
                    pass
            update_movie(state['key'], '转换中', f'stream {track["index"]}（{track["title"] or track["language"]}）OCR{detail}')

        adapter = CODE_DIR / ('ocr_tesseract.py' if track['model'].get('engine') == 'tesseract' else 'paddle_batch.py')
        if not adapter.exists():
            raise RuntimeError(f'OCR适配器尚未准备好：{adapter}')
        if not complete_json:
            run_command([PYTHON, adapter, 'ocr', '-i', Path(track['bdn']).parent,
                         '--lang', track['paddle_language'], '--save_path', json_dir],
                        raw.parent / 'ocr.log', env=env_for(track), progress=heartbeat)
            if recognition_identity(track, events)[1] != fingerprint:
                raise RuntimeError('识别期间模型、输入或适配器发生变化，拒绝采用结果')
            track['recognition_proof'] = verify_completed_ocr(track, events, identity)
            save(state)
        write_srt(raw, read_ocr_entries(track, events))
    report = validate_ocr_srt(raw, track['baseline'], track['duration'],
                              raise_on_error=False, **nontext_options(track))
    fatal = [error for error in report['errors'] if error.get('kind') != 'empty_text']
    track['pre_review_validation'] = report
    if fatal:
        raise RuntimeError(f'OCR时间码或结构校验失败：{fatal[:8]}')
    track['empty_text_indices'] = [e['index'] for e in report['errors'] if e.get('kind') == 'empty_text']
    entries = parse_srt(raw)
    prepared_candidate = candidate_entries(track, entries)
    candidate = Path(track['candidate_srt'])
    if candidate.exists():
        expected = [(e['start_ms'], e['end_ms'], e['text']) for e in prepared_candidate]
        actual = [(e['start_ms'], e['end_ms'], e['text']) for e in parse_srt(candidate)]
        if actual != expected:
            # Only regenerate this task's previously recorded candidate; a
            # manually supplied new file without provenance is never overwritten.
            if not track.get('candidate_sha256') or digest(candidate) != track['candidate_sha256']:
                raise RuntimeError(f'候选SRT发生未记录编辑，请先检查：{candidate}')
            write_srt(candidate, prepared_candidate, overwrite=True)
    else:
        write_srt(candidate, prepared_candidate)
    track.pop('error', None)
    track.update(status='prepared', raw_sha256=digest(raw), candidate_sha256=digest(candidate))
    save(state)
    log(f'{state["title"]} / stream {track["index"]}: {len(entries)} 条，待图文核对；空文本 {len(track["empty_text_indices"])}')


def call_review(state, args):
    state_path = Path(state['work_dir']) / 'state.json'
    review_path = state_path.with_name('review.json')
    if args.prepare_only:
        return None
    command = args.review_command or os.environ.get('PGS_REVIEW_COMMAND')
    if command:
        run_command(shlex.split(command) + [str(state_path)], state_path.with_name('review-command.log'))
    else:
        try:
            module = importlib.import_module('visual_review')
        except ModuleNotFoundError as exc:
            if exc.name != 'visual_review':
                raise
            return json.loads(review_path.read_text()) if review_path.exists() else None
        result = module.review_movie(state_path)
        if isinstance(result, dict) and not review_path.exists():
            atomic_write_json(review_path, result)
    return json.loads(review_path.read_text()) if review_path.exists() else None


def strict_pre_publish(state):
    assert_unchanged(state)
    if not state.get('source_scan_complete') or not state['tracks'] or set(state['tracks']) != set(state.get('expected_track_ids', [])):
        raise RuntimeError('源文件/轨道覆盖不完整，禁止发布')
    for track in state['tracks'].values():
        if track.get('duplicate_of'):
            representative = state['tracks'].get(track['duplicate_of'])
            if representative is None or representative.get('duplicate_of'):
                raise RuntimeError('去重代表关系不完整')
            continue
        identity, fingerprint = recognition_identity(track, read_bdn(track['bdn'], track['baseline']))
        if track.get('recognition_fingerprint') != fingerprint:
            raise RuntimeError(f'发布前识别配置/实际模型或输入签名已改变：{track["id"]}')
        report = validate_ocr_srt(track['raw_srt'], track['baseline'], track['duration'],
                                  **nontext_options(track))
        entries = parse_srt(track['raw_srt'])
        candidate = parse_srt(track['candidate_srt'])
        if [(e['start_ms'], e['end_ms'], e['text']) for e in candidate] != [
                (e['start_ms'], e['end_ms'], e['text']) for e in candidate_entries(track, entries)]:
            raise RuntimeError(f'成品与已核对原始文本不一致：{track["id"]}')
        track.update(raw_sha256=digest(track['raw_srt']),
                     candidate_sha256=digest(track['candidate_srt']), final_entries=len(candidate),
                     validation_coverage=report['coverage'])


def publish(state, review):
    if not review or review.get('approved') is not True:
        raise RuntimeError('图文核对未批准，不能发布或标记完成')
    strict_pre_publish(state)
    if review.get('movie_key') != state['key']:
        raise RuntimeError('图文核对批准未明确绑定当前电影')
    # Check every destination before creating any, so a collision cannot silently
    # substitute unrelated or hand-corrected subtitles for this run's outputs.
    unique = [t for t in state['tracks'].values() if not t.get('duplicate_of')]
    if (not isinstance(review.get('tracks'), list) or len(review['tracks']) != len(unique)
            or any(not isinstance(item, dict) or not isinstance(item.get('track_id'), str) for item in review['tracks'])):
        raise RuntimeError('图文批准必须逐项列出所有独立字幕版本')
    approved_tracks = {item['track_id']: item for item in review['tracks']}
    if len(approved_tracks) != len(unique) or set(approved_tracks) != {track['id'] for track in unique}:
        raise RuntimeError('图文批准未唯一覆盖所有独立字幕版本')
    for track in unique:
        approved = approved_tracks[track['id']]
        if approved.get('approved') is not True or approved.get('raw_sha256') != track['raw_sha256'] or approved.get('candidate_sha256') != track['candidate_sha256']:
            raise RuntimeError(f'字幕已改变或未通过图文批准：{track["id"]}')
    targets = [t['output'] for t in unique]
    if len(set(targets)) != len(targets):
        raise RuntimeError('不同字幕版本的输出文件名发生冲突')
    for track in unique:
        target = Path(track['output'])
        if target.exists():
            owned_hash = track.get('published_sha256') or (track.get('publication_intent') or {}).get('sha256')
            if not owned_hash or digest(target) != owned_hash:
                raise FileExistsError(f'拒绝覆盖已有字幕：{target}')
    save(state)
    for track in unique:
        target = Path(track['output'])
        if not target.exists():
            track['publication_intent'] = {'path': str(target), 'sha256': track['candidate_sha256']}
            save(state)
            with Path(track['candidate_srt']).open('rb') as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        if digest(target) != track['candidate_sha256']:
            raise RuntimeError(f'发布后字幕校验值不符：{target}')
        track['published_sha256'] = track['candidate_sha256']
        track['status'] = 'published'
        save(state)
    assert_unchanged(state)
    state.update(status='completed', completed_at=now(), review=review)
    save(state)
    note = (f'已发布 {len(unique)} 份 SRT，PGS {len(state["tracks"])} 轨；'
            f'全部毫秒时间/非空校验通过；{review.get("reviewer", "图文抽查")}：{review.get("notes", "通过")}；'
            f'记录：{state["work_dir"]}/state.json')
    nontext_count = sum(track.get('validation_coverage', {}).get('reviewed_nontext_events', 0)
                        for track in unique)
    if nontext_count:
        note = note.replace('全部毫秒时间/非空校验通过',
                            f'全部原始事件保留且时间精确，文本项非空；{nontext_count} 项经确认无文字，按绑定记录从成品省略')
    update_movie(state['key'], '已完成', note, completed=True)
    log(f'完成 {state["title"]}：{len(unique)} 份 SRT')


def cleanup_bounded_export(track, local):
    """Remove owned fragment SUP/PNG data while retaining its timing/audit logs."""
    expected = local(track['bdn'])
    allowed_baselines = {track['baseline']: digest(track['baseline'])}
    allowed_sources = {track['sup']: track.get('sup_sha256')}
    for staging in expected.parent.parent.glob('.bounded-bdn-*'):
        if not re.fullmatch(r'\.bounded-bdn-[0-9a-f]{32}', staging.name):
            continue
        staging = local(staging)
        audit_path = staging / 'audit.json'
        audit = json.loads(audit_path.read_text())
        source = audit.get('source', {})
        baseline = audit.get('baseline', {})
        if (audit.get('schema') != 'bounded-bdn-v1'
                or audit.get('expected_xml') != str(expected)
                or source.get('path') not in allowed_sources
                or not allowed_sources[source['path']]
                or source.get('sha256') != allowed_sources[source['path']]
                or baseline.get('path') not in allowed_baselines
                or baseline.get('sha256') != allowed_baselines[baseline['path']]):
            raise RuntimeError(f'分批缓存归属校验失败，保留文件：{staging}')
        for number, _batch in enumerate(audit['plan']['batches'], 1):
            name = f'batch-{number:04d}'
            fragment = local(staging / name / f'{name}.sup')
            images = local(staging / name / 'images' / name)
            if images.exists():
                shutil.rmtree(images)
            fragment.unlink(missing_ok=True)
        merged = local(staging / 'merged')
        if merged.exists():
            shutil.rmtree(merged)
        audit.update(reproducible_intermediates_removed=True, cleanup_completed_at=now())
        atomic_write_json(audit_path, audit)


def cleanup_completed(state):
    """Archive OCR evidence, verify it, then remove reproducible large inputs."""
    if state.get('status') != 'completed':
        return
    directory = Path(state['work_dir']).resolve()
    review_rows = {row['track_id']: row for row in state['review'].get('tracks', [])}

    def local(path):
        path = Path(path)
        if not path.resolve().is_relative_to(directory):
            raise RuntimeError(f'清理路径越出本电影工作目录：{path}')
        return path

    def verify_archive(path):
        with tarfile.open(path, 'r:gz') as archive:
            manifest = json.load(archive.extractfile('MANIFEST.json'))
            expected = {row['path']: row for row in manifest['files']}
            members = {m.name: m for m in archive.getmembers()}
            if set(members) != set(expected) | {'MANIFEST.json'}:
                raise RuntimeError('OCR归档成员与清单不一致')
            for name, row in expected.items():
                member = members[name]
                if not member.isfile() or member.size != row['size']:
                    raise RuntimeError('OCR归档成员类型或长度不符')
                h = hashlib.sha256()
                with archive.extractfile(member) as content:
                    for block in iter(lambda: content.read(1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != row['sha256']:
                    raise RuntimeError('OCR归档成员校验值不符：' + name)
        return manifest

    for track in state['tracks'].values():
        if not track.get('duplicate_of'):
            # Every image actually reviewed, including all low-score/empty
            # cases, must survive at full resolution in visual-review/.
            row = review_rows.get(track['id'])
            selected = row.get('requested_indices', []) if row else []
            if not row or row.get('approved') is not True or not selected or not track.get('visual_review_directory'):
                raise RuntimeError(f'缺少该轨校对原图的保留记录，保留缓存：{track["id"]}')
            review_dir = local(track['visual_review_directory'])
            bdn = local(track['bdn'])
            if bdn.exists():
                events = read_bdn(bdn, track['baseline'])
                for index in selected:
                    source = Path(events[index - 1]['image_path'])
                    preserved = review_dir / 'original-images' / f'{index:06d}{source.suffix}'
                    if not preserved.is_file() or digest(source) != digest(preserved):
                        raise RuntimeError(f'抽查原图尚未完整保留：{track["id"]} #{index}')
            json_dir = local(track['ocr_json_dir'])
            evidence = local(track['evidence_dir'])
            archive_dir = directory / 'ocr-archives'
            archive_dir.mkdir(exist_ok=True)
            archive_path = archive_dir / (safe_label(track['id']) + '.tar.gz')
            archived = track.get('ocr_archive')
            if archive_path.exists():
                if not archived or archived.get('sha256') != digest(archive_path):
                    raise RuntimeError(f'已有未知或已改变的OCR归档：{archive_path}')
                verify_archive(archive_path)
            else:
                files = set(p for p in json_dir.rglob('*') if p.is_file() and not p.is_symlink())
                for p in evidence.rglob('*'):
                    parts = p.relative_to(evidence).parts
                    if (p.is_file() and not p.is_symlink()
                            and not any(part in {'original', 'prepared', 'paddle-cache', 'tessdata',
                                                 'language-identification'} for part in parts)
                            and p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.lock'}):
                        files.add(p)
                records = [{'path': p.relative_to(directory).as_posix(),
                            'size': p.stat().st_size, 'sha256': digest(p)} for p in sorted(files)]
                if not records or not any(r['path'].endswith('_res.json') for r in records):
                    raise RuntimeError('没有完整OCR JSON可以归档')
                manifest = {'track_id': track['id'], 'movie_key': state['key'],
                            'created_at': now(), 'files': records}
                import io
                payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
                fd, temporary_name = tempfile.mkstemp(prefix='ocr-evidence-', suffix='.tar.gz.partial', dir=archive_dir)
                os.close(fd)
                temporary = Path(temporary_name)
                try:
                    with tarfile.open(temporary, 'w:gz', compresslevel=6) as archive:
                        info = tarfile.TarInfo('MANIFEST.json')
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
                        for item in records:
                            archive.add(directory / item['path'], arcname=item['path'], recursive=False)
                    verify_archive(temporary)
                    with temporary.open('rb') as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary, archive_path)
                finally:
                    temporary.unlink(missing_ok=True)
                track['ocr_archive'] = {'path': str(archive_path), 'sha256': digest(archive_path),
                                        'files': len(records), 'verified_at': now()}
                save(state)
            # The archive has been read back and each member's SHA-256 checked.
            # Keep language-identification originals and the separate complete
            # visual-review record; prune only this adapter's known work paths.
            for path in (json_dir, evidence / 'paddle-evidence', evidence / 'paddle-cache'):
                path = local(path)
                if path.exists():
                    shutil.rmtree(path)
            for name in ('paddle-progress.json', 'paddle-flags.json'):
                local(evidence / name).unlink(missing_ok=True)
        sup = local(track['sup'])
        image_dir = local(track['bdn']).parent
        cleanup_bounded_export(track, local)
        # Timing baselines, all raw/unreviewed/corrected/final SRTs and full
        # selected review originals/montages remain outside these paths.
        for path in (sup, image_dir):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    state['reproducible_intermediates_removed'] = True
    state['cleanup_completed_at'] = now()
    save(state)


def process_movie(movie, args, media_root):
    state = state_for(movie, args.resume)
    state['errors'] = []
    save(state)
    log(f'开始 {movie["key"]}')
    if not args.finalize_only:
        update_movie(movie['key'], '转换中', f'串行批次；记录：{state["work_dir"]}/state.json')
        prepare_sources(movie, state, media_root)
        representatives = {}
        for track in state['tracks'].values():
            try:
                prepare_track(state, track, representatives, offline=args.offline)
            except BatchStop:
                raise
            except Exception as exc:
                track.update(status='error', error=str(exc))
                state['errors'].append({'track': track['id'], 'error': str(exc)})
                save(state)
                log(f'该轨暂未完成：{track["id"]}：{exc}')
    if (not state.get('source_scan_complete') or not state['tracks'] or state['errors']
            or any(t.get('status') not in {'prepared', 'duplicate', 'published'} for t in state['tracks'].values())):
        state['status'] = 'needs_attention'
        save(state)
        update_movie(movie['key'], '待处理异常', f'部分轨道尚未完成；详情：{state["work_dir"]}/state.json')
        return state
    state['status'] = 'awaiting_review'
    save(state)
    update_movie(movie['key'], '待校对', f'全部轨道已准备，正在逐轨图文核对；记录：{state["work_dir"]}/state.json')
    review = call_review(state, args)
    # The reviewer may correct raw captions and update candidate hashes in state.
    state = json.loads((Path(state['work_dir']) / 'state.json').read_text())
    if not review or review.get('approved') is not True:
        state['status'] = 'awaiting_review'
        save(state)
        update_movie(movie['key'], '待校对', f'尚未通过全部图文核对；{(review or {}).get("notes", "等待核对程序")}；记录：{state["work_dir"]}/state.json')
        return state
    publish(state, review)
    if args.cleanup:
        try:
            cleanup_completed(state)
        except Exception as exc:
            state['cleanup_error'] = str(exc)
            save(state)
            log(f'电影已完成，临时证据清理异常，保留原文件：{exc}')
    return state


def record_failure(key, status, note):
    """A post-publication maintenance error must never reopen a completed movie."""
    current = next(item for item in parse_movies() if item['key'] == key)
    if current['completed']:
        update_movie(key, '已完成', '字幕已完成；后续维护记录：' + str(note), completed=True)
        return True
    update_movie(key, status, note)
    return False


def prioritize_movies(selected, priority_keys, known_keys):
    """Move explicitly requested unfinished movies first, retaining stable order."""
    if len(priority_keys) != len(set(priority_keys)):
        raise ValueError('优先电影重复')
    missing = set(priority_keys) - set(known_keys)
    if missing:
        raise ValueError('优先电影不在清单中：' + '；'.join(sorted(missing)))
    order = {key: index for index, key in enumerate(priority_keys)}
    return sorted(selected, key=lambda movie: order.get(movie['key'], len(order)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-root', default=os.environ.get('PGS_MEDIA_ROOT'),
                        help='Root directory containing the movie folders in the manifest')
    parser.add_argument('--offline', action='store_true',
                        help='Use only local OCR; requires --prepare-only and known languages')
    parser.add_argument('--only', help='Only this movie key, e.g. jellyfin/片名.English.2025')
    parser.add_argument('--prioritize', action='append', default=[],
                        help='Process this movie first, then the remaining serial queue; repeat for ordered priorities')
    parser.add_argument('--limit', type=int, help='Maximum number of uncompleted movies to process')
    parser.add_argument('--resume', action='store_true', help='Reuse this batch\'s existing state and image cache')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare candidates, without review or publication')
    parser.add_argument('--finalize-only', action='store_true', help='Review/publish prepared state without extraction or OCR')
    parser.add_argument('--review-command', help='Argument string for synchronous reviewer; state.json is appended, no shell')
    parser.add_argument('--cleanup', action='store_true', help='Remove reproducible SUP/BDN images only after completion')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.prepare_only and args.finalize_only:
        parser.error('--prepare-only and --finalize-only are mutually exclusive')
    if args.finalize_only:
        args.resume = True
    if args.offline and not args.prepare_only:
        parser.error('--offline requires --prepare-only')
    if not args.media_root or not Path(args.media_root).expanduser().is_dir():
        parser.error('set --media-root or PGS_MEDIA_ROOT to an existing directory')
    media_root = Path(args.media_root).expanduser().resolve()
    if not SECONV or not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        parser.error('install FFmpeg (and ffprobe) and SeConv, or set PGS_SECONV')
    if TMP_ROOT.is_relative_to('/tmp') or WORK == TMP_ROOT or not WORK.is_relative_to(TMP_ROOT):
        parser.error('set PGS_WORK_DIR to a subdirectory of TMPDIR outside /tmp')
    if not MANIFEST.is_file():
        parser.error(f'create the queue manifest or set PGS_MANIFEST: {MANIFEST}')
    WORK.mkdir(parents=True, exist_ok=True)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt('收到停止信号；保留当前状态以便恢复')
    signal.signal(signal.SIGTERM, terminate)
    with batch_lock():
        movies = parse_movies()
        if args.only and args.only not in {m['key'] for m in movies}:
            raise RuntimeError(f'清单未找到电影：{args.only}')
        waiting_input = [m['key'] for m in movies if not m['completed'] and m['status'] == '待用户确认']
        selected = [m for m in movies if not m['completed']
                    and (not args.only or m['key'] == args.only)
                    and (args.only or m['status'] != '待用户确认')]
        selected = prioritize_movies(selected, args.prioritize, [m['key'] for m in movies])
        if waiting_input and not args.only:
            log('保留待用户确认项目，继续其余电影：' + '；'.join(waiting_input))
        if args.limit:
            selected = selected[:args.limit]
        results = []
        for movie in selected:
            try:
                check_space()
                state = process_movie(movie, args, media_root)
                results.append({'key': movie['key'], 'status': state['status']})
            except (BatchStop, KeyboardInterrupt) as exc:
                record_failure(movie['key'], '已中断，可恢复', str(exc))
                log(f'批次停止：{exc}')
                raise
            except Exception as exc:
                completed = record_failure(movie['key'], '待处理异常', f'{str(exc)[:1000]}；工作目录：{work_for(movie)}')
                log(f'{"已完成电影的后续维护异常" if completed else "电影尚未完成"}：{movie["key"]}：{exc}')
                results.append({'key': movie['key'], 'status': 'completed' if completed else 'error', 'error': str(exc)})
            atomic_write_json(WORK / 'queue-status.json', {'updated_at': now(), 'pid': os.getpid(),
                                                         'last_key': movie['key'], 'results': results,
                                                         'waiting_user_input': waiting_input})
        log(f'本次串行队列结束：{len(results)} 部，完成 {sum(r["status"] == "completed" for r in results)} 部')

        accepted = {'completed', 'awaiting_review'} if args.prepare_only else {'completed'}
        return int(any(row['status'] not in accepted for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
