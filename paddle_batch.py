#!/usr/bin/env python3
"""Task-local, sequential PaddleOCR CLI adapter for seconv 5.3.0."""
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

from pgs_config import MODEL_ROOT


SUPPORTED_LANGS = set('ch chinese_cht en japan af az bs cs cy da de es et fr ga hr hu id is it ku la lt lv mi ms mt nl no oc pl pt ro rs_latin sk sl sq sv sw tl tr uz vi french german fi eu gl lb rm ca qu'.split())
V5_MODEL_LANGS = {
    'PP-OCRv5_server_rec': {'ch', 'chinese_cht', 'japan'},
    'en_PP-OCRv5_mobile_rec': {'en'},
    'latin_PP-OCRv5_mobile_rec': (SUPPORTED_LANGS - {'ch', 'chinese_cht', 'en', 'japan'}) | {'pi'},
    'eslav_PP-OCRv5_mobile_rec': {'ru', 'be', 'uk'},
    'cyrillic_PP-OCRv5_mobile_rec': set('ru rs_cyrillic be bg uk mn abq ady kbd ava dar inh che lbe lez tab kk ky tg mk tt cv ba mhr mo udm kv os bua xal tyv sah kaa'.split()),
    'arabic_PP-OCRv5_mobile_rec': {'ar', 'fa', 'ug', 'ur', 'ps', 'ku', 'sd', 'bal'},
    'devanagari_PP-OCRv5_mobile_rec': set('hi mr ne bh mai ang bho mah sck new gom sa bgc'.split()),
    'korean_PP-OCRv5_mobile_rec': {'korean'},
    'th_PP-OCRv5_mobile_rec': {'th'},
    'el_PP-OCRv5_mobile_rec': {'el'},
    'te_PP-OCRv5_mobile_rec': {'te'},
    'ta_PP-OCRv5_mobile_rec': {'ta'},
}
SCRIPT_RANGES = {
    'eslav': (0x0400, 0x052F), 'cyrillic': (0x0400, 0x052F),
    'arabic': (0x0600, 0x06FF), 'devanagari': (0x0900, 0x097F),
    'korean': (0xAC00, 0xD7AF), 'th': (0x0E00, 0x0E7F),
    'el': (0x0370, 0x03FF), 'te': (0x0C00, 0x0C7F), 'ta': (0x0B80, 0x0BFF),
}
SETTINGS = {
    'adapter': 'pgs-paddle-v1', 'preprocess': 'verified-10-transparent-10-black-crop-v1',
    'engine': 'onnxruntime', 'device': 'cpu', 'intra_op_num_threads': 2,
    'inter_op_num_threads': 1, 'execution_mode': 'sequential', 'allow_spinning': False,
    'det_limit_side_len': 64, 'det_limit_type': 'min', 'det_max_side_limit': 4000,
    'det_thresh': 0.3, 'det_box_thresh': 0.6, 'det_unclip_ratio': 1.5,
    'rec_score_thresh': 0.0, 'rec_batch_size': 6, 'low_score_threshold': 0.90,
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pgs-json-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    if sys.argv[1:] == ['--version']:
        print(importlib.metadata.version('paddleocr'))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ocr'])
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--lang', default='ch')
    parser.add_argument('--ocr_version', default=None, choices=['PP-OCRv6', 'PP-OCRv5'])
    parser.add_argument('--device', default='cpu', choices=['cpu'])
    parser.add_argument('--engine', default='onnxruntime', choices=['onnxruntime'])
    for name in ('use_doc_orientation_classify', 'use_doc_unwarping', 'use_textline_orientation', 'use_angle_cls'):
        parser.add_argument('--' + name, default='False', choices=['False', 'false'])
    args = parser.parse_args()
    override_name = os.environ.get('PGS_REC_MODEL_NAME')
    override_dir = os.environ.get('PGS_REC_MODEL_DIR')
    if bool(override_name) != bool(override_dir):
        parser.error('PGS_REC_MODEL_NAME and PGS_REC_MODEL_DIR must be supplied together')
    args.rec_model_name = override_name or 'PP-OCRv6_medium_rec'
    args.rec_model_dir = override_dir or str(MODEL_ROOT / 'PP-OCRv6_medium_rec_onnx')
    supported = SUPPORTED_LANGS if args.rec_model_name == 'PP-OCRv6_medium_rec' else V5_MODEL_LANGS.get(args.rec_model_name)
    if supported is None or args.lang not in supported:
        parser.error(f'Recognizer {args.rec_model_name!r} is not explicitly supported for language {args.lang!r}; no fallback is performed')
    effective_version = 'PP-OCRv6' if args.rec_model_name == 'PP-OCRv6_medium_rec' else 'PP-OCRv5'
    if args.ocr_version is not None and args.ocr_version != effective_version:
        parser.error('--ocr_version conflicts with the explicitly selected recognition model')
    args.ocr_version = effective_version
    if not os.environ.get('PGS_TRACK_WORK'):
        parser.error('PGS_TRACK_WORK must identify this track\'s evidence directory under $TMPDIR')
    return args


def validate_recognizer(args):
    """Validate local model identity and alphabet before any inference or cache use."""
    import yaml
    directory = Path(args.rec_model_dir)
    if not directory.is_absolute():
        raise ValueError('PGS_REC_MODEL_DIR must be an absolute local directory')
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('Recognition model path is not a directory: ' + str(directory))
    model = directory / 'inference.onnx'
    config_file = directory / 'inference.yml'
    if not model.is_file() or model.stat().st_size < 32 or not config_file.is_file():
        raise ValueError('Recognition model requires nonempty inference.onnx and inference.yml')
    config = yaml.safe_load(config_file.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('Global', {}).get('model_name') != args.rec_model_name:
        raise ValueError('inference.yml Global.model_name does not match PGS_REC_MODEL_NAME')
    post = config.get('PostProcess', {})
    characters = post.get('character_dict')
    if post.get('name') != 'CTCLabelDecode' or not isinstance(characters, list) or not characters or any(not isinstance(character, str) or not character for character in characters):
        raise ValueError('Recognition model requires a valid embedded CTC character dictionary')
    script = SCRIPT_RANGES.get(args.rec_model_name.split('_', 1)[0])
    if script and not any(script[0] <= ord(character) <= script[1] for entry in characters for character in entry):
        raise ValueError('Recognition dictionary contains no characters for its declared script')
    dictionary_sha256 = hashlib.sha256(json.dumps(characters, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    expected = os.environ.get('PGS_REC_DICT_SHA256')
    if expected is not None and expected.lower() != dictionary_sha256:
        raise ValueError('Recognition dictionary SHA256 does not match PGS_REC_DICT_SHA256')
    args.rec_model_dir = str(directory)
    return {'model_name': args.rec_model_name, 'model_dir': str(directory),
            'ocr_version': args.ocr_version, 'character_count': len(characters),
            'dictionary_sha256': dictionary_sha256}


def prepare_image(image):
    from PIL import Image, ImageChops, ImageOps
    rgba = image.convert('RGBA')
    width, height = rgba.size
    info = {'method': 'unchanged', 'original_size': [width, height]}
    if width <= 40 or height <= 40:
        return rgba, info
    outer = [(0, 0, width, 10), (0, height - 10, width, height),
             (0, 10, 10, height - 10), (width - 10, 10, width, height - 10)]
    ring = [(10, 10, width - 10, 20), (10, height - 20, width - 10, height - 10),
            (10, 20, 20, height - 20), (width - 20, 20, width - 10, height - 20)]
    transparent = all(rgba.crop(box).getchannel('A').getextrema() == (0, 0) for box in outer)
    black = all(rgba.crop(box).getextrema() == ((0, 0), (0, 0), (0, 0), (255, 255)) for box in ring)
    inner = rgba.crop((20, 20, width - 20, height - 20))
    if not (transparent and black and inner.getchannel('A').getextrema() == (255, 255)):
        return rgba, info
    rgb = inner.convert('RGB')
    box = ImageChops.difference(rgb, Image.new('RGB', rgb.size, 'white')).getbbox()
    if box is None:
        return rgba, info
    prepared = ImageOps.expand(rgb.crop(box), border=16, fill='white')
    info.update(method='verified-seconv-border-and-white-crop', crop_in_original=[box[0] + 20, box[1] + 20, box[2] + 20, box[3] + 20], added_border=16)
    return prepared, info


def create_model(args):
    import onnxruntime as ort
    original_init = ort.InferenceSession.__init__

    def configured_init(instance, path_or_bytes, sess_options=None, *args, **kwargs):
        options = sess_options if sess_options is not None else ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        return original_init(instance, path_or_bytes, options, *args, **kwargs)

    ort.InferenceSession.__init__ = configured_init
    from paddleocr import PaddleOCR
    from paddlex.inference import load_pipeline_config
    config = load_pipeline_config('OCR')
    for name in ('TextDetection', 'TextRecognition'):
        config['SubModules'][name]['engine_config'] = {
            'intra_op_num_threads': 2, 'inter_op_num_threads': 1, 'execution_mode': 'sequential',
        }
    config['SubModules']['TextDetection']['max_side_limit'] = 4000
    return PaddleOCR(
        text_detection_model_name='PP-OCRv6_medium_det',
        text_detection_model_dir=str(MODEL_ROOT / 'PP-OCRv6_medium_det_onnx'),
        text_recognition_model_name=args.rec_model_name,
        text_recognition_model_dir=args.rec_model_dir,
        engine='onnxruntime', device='cpu', cpu_threads=2,
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, text_det_limit_side_len=64,
        text_det_limit_type='min', text_det_thresh=0.3, text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.5, text_rec_score_thresh=0.0,
        text_recognition_batch_size=6, paddlex_config=config,
    )


def run(args):
    from PIL import Image
    recognizer = validate_recognizer(args)
    source = Path(args.input).resolve(strict=True)
    output = Path(args.save_path).resolve()
    track = Path(os.environ['PGS_TRACK_WORK']).resolve()
    if not track.is_absolute() or str(track) == '/tmp' or str(track).startswith('/tmp/'):
        raise ValueError('PGS_TRACK_WORK must use the authorized $TMPDIR, never /tmp')
    track.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    images = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in suffixes), key=lambda p: p.name) if source.is_dir() else [source]
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError('No input images, or duplicate image basenames')
    lock = (track / 'paddle.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_id = time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    evidence = track / 'paddle-evidence' / run_id
    for folder in ('original', 'prepared', 'results'):
        (evidence / folder).mkdir(parents=True, exist_ok=False)
    cache = track / 'paddle-cache'
    cache.mkdir(exist_ok=True)
    signature_data = {'settings': SETTINGS, 'language': args.lang, 'versions': {}, 'models': {}, 'recognizer': recognizer}
    for package in ('paddleocr', 'paddlex', 'onnxruntime', 'Pillow'):
        signature_data['versions'][package] = importlib.metadata.version(package)
    for role, directory in (('detector', MODEL_ROOT / 'PP-OCRv6_medium_det_onnx'), ('recognizer', Path(args.rec_model_dir))):
        # Include config, weights and any external weight files in the cache identity.
        for path in sorted(directory.rglob('*')):
            relative_path = path.relative_to(directory)
            if path.is_file() and not any(part.startswith('.') for part in relative_path.parts):
                signature_data['models'][role + '/' + relative_path.as_posix()] = digest_file(path)
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    write_json(evidence / 'configuration.json', signature_data)
    state = {'status': 'preparing', 'run_id': run_id, 'evidence': str(evidence), 'language': args.lang,
             'total': len(images), 'completed': 0, 'cache_hits': 0, 'inferences': 0,
             'flagged': 0, 'started_at': time.time(), 'signature': signature}
    mapping = []
    flags = []
    model = None

    def checkpoint(status):
        state.update(status=status, updated_at=time.time(), elapsed_seconds=round(time.time() - state['started_at'], 2))
        write_json(track / 'paddle-progress.json', state)
        write_json(evidence / 'progress.json', state)

    checkpoint('preparing')
    try:
        for index, path in enumerate(images):
            original = evidence / 'original' / path.name
            original.write_bytes(path.read_bytes())
            with Image.open(original) as image:
                image.load()
                prepared, transform = prepare_image(image)
            image_key = hashlib.sha256(signature.encode() + repr((prepared.mode, prepared.size)).encode() + prepared.tobytes()).hexdigest()
            prepared_path = evidence / 'prepared' / (path.stem + '.png')
            prepared.save(prepared_path)
            cached = cache / (image_key + '.json')
            result = None
            if cached.exists():
                saved = json.loads(cached.read_text(encoding='utf-8'))
                if saved.get('signature') != signature or saved.get('image_key') != image_key:
                    raise ValueError('OCR cache signature mismatch: ' + str(cached))
                result = copy.deepcopy(saved['result'])
                state['cache_hits'] += 1
            if result is None:
                if model is None:
                    checkpoint('loading_model')
                    model = create_model(args)
                    checkpoint('recognizing')
                results = model.predict(str(prepared_path))
                if len(results) != 1:
                    raise ValueError(f'Expected one OCR result for {path.name}, got {len(results)}')
                result = results[0].json['res']
                if not all(key in result for key in ('rec_texts', 'rec_scores', 'rec_polys', 'rec_boxes')):
                    raise ValueError('OCR result is missing required recognition fields')
                write_json(cached, {'signature': signature, 'image_key': image_key, 'result': result})
                state['inferences'] += 1
            result['input_path'] = str(prepared_path)
            texts = result['rec_texts']
            scores = result['rec_scores']
            if len(texts) != len(scores):
                raise ValueError('Text and confidence counts differ for ' + path.name)
            name = path.stem + '_res.json'
            write_json(evidence / 'results' / name, result)
            write_json(output / name, result)
            item = {'index': index, 'basename': path.stem, 'source': str(path), 'original': str(original),
                    'original_sha256': digest_file(original), 'prepared': str(prepared_path),
                    'image_key': image_key, 'transform': transform, 'result': str(evidence / 'results' / name)}
            mapping.append(item)
            if not any(text.strip() for text in texts) or any(float(score) < 0.90 for score in scores):
                flags.append({'index': index, 'basename': path.stem, 'texts': texts, 'scores': scores,
                              'original': str(original), 'result': item['result']})
            state.update(completed=index + 1, flagged=len(flags), latest_basename=path.stem)
            # Append each mapping durably; the JSON results are also atomic checkpoints.
            with (evidence / 'mapping.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            checkpoint('recognizing')
            if (index + 1) % 25 == 0 or index + 1 == len(images):
                print(json.dumps({'pgs_ocr_progress': state}, ensure_ascii=False), flush=True)
        write_json(evidence / 'flags.json', flags)
        write_json(track / 'paddle-flags.json', {'run_id': run_id, 'flags': flags})
        checkpoint('completed')
    except Exception as exc:
        state['error'] = str(exc)
        write_json(evidence / 'flags.json', flags)
        checkpoint('failed')
        raise
    finally:
        if model is not None:
            model.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    run(parse_args())
