#!/usr/bin/env python3
"""Sequential Hebrew fallback, with the same evidence/result protocol as Paddle."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from paddle_batch import write_json, digest_file
from pgs_config import TESSDATA_ENG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ocr'])
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('--lang', required=True, choices=['he', 'heb'])
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()
    from PIL import Image, ImageOps
    root = Path(os.environ['TMPDIR']).resolve()
    track = Path(os.environ['PGS_TRACK_WORK']).resolve()
    output = Path(args.save_path).resolve()
    if not track.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError('OCR evidence and results must stay inside TMPDIR')
    source = Path(args.input)
    images = sorted(source.glob('*.png')) if source.is_dir() else [source]
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError('Missing images or duplicate basenames')
    model_dir = Path(os.environ['PGS_TESSDATA_DIR'])
    heb = model_dir / 'heb.traineddata'
    eng = TESSDATA_ENG
    for model in (heb, eng):
        if not model.is_file() or not model.stat().st_size:
            raise FileNotFoundError(model)
    track.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    lock = (track / 'paddle.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_id = time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    evidence = track / 'paddle-evidence' / run_id
    for name in ('original', 'prepared', 'results', 'tessdata'):
        (evidence / name).mkdir(parents=True)
    for model in (heb, eng):
        (evidence / 'tessdata' / model.name).symlink_to(model)
    config = {'engine': 'tesseract', 'language': 'heb+eng', 'psm': 6,
              'preprocess': 'RGBA-on-black-alpha-bbox-grayscale-invert-white-border24-v1',
              'version': subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0],
              'models': {p.name: digest_file(p) for p in (heb, eng)}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write_json(evidence / 'configuration.json', config)
    state = {'status': 'preparing', 'run_id': run_id, 'evidence': str(evidence),
             'language': 'heb', 'engine': 'tesseract', 'total': len(images),
             'completed': 0, 'flagged': 0, 'started_at': time.time(), 'signature': signature}
    def checkpoint(status):
        state.update(status=status, updated_at=time.time(), elapsed_seconds=round(time.time()-state['started_at'], 2))
        write_json(track / 'paddle-progress.json', state)
        write_json(evidence / 'progress.json', state)
    try:
        checkpoint('preparing')
        prepared_paths = []
        for index, path in enumerate(images, 1):
            original = evidence / 'original' / path.name
            shutil.copyfile(path, original)
            with Image.open(original) as image:
                rgba = image.convert('RGBA')
                bbox = rgba.getchannel('A').getbbox()
                if bbox is not None:
                    rgba = rgba.crop(bbox)
                background = Image.new('RGBA', rgba.size, (0, 0, 0, 255))
                background.alpha_composite(rgba)
                prepared = ImageOps.expand(ImageOps.invert(background.convert('L')), border=24, fill=255)
                dest = evidence / 'prepared' / f'{index:06d}.png'
                prepared.save(dest)
            prepared_paths.append(dest)
        imagelist = evidence / 'images.txt'
        imagelist.write_text(''.join(str(p)+'\n' for p in prepared_paths), encoding='utf-8')
        output_base = evidence / 'tesseract'
        env = os.environ.copy()
        env['OMP_THREAD_LIMIT'] = '2'
        checkpoint('recognizing')
        with (evidence / 'tesseract.log').open('w') as log:
            proc = subprocess.run(['tesseract', str(imagelist), str(output_base),
                '--tessdata-dir', str(evidence / 'tessdata'), '-l', 'heb+eng',
                '--psm', '6', '--dpi', '150', '-c', 'tessedit_create_txt=1',
                '-c', 'tessedit_create_tsv=1'], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                timeout=86400)
        if proc.returncode:
            raise RuntimeError(f'Tesseract exited {proc.returncode}; see {evidence / "tesseract.log"}')
        pages = output_base.with_suffix('.txt').read_text(encoding='utf-8').split('\f')
        if pages and not pages[-1].strip():
            pages.pop()
        if len(pages) != len(images):
            raise ValueError(f'Tesseract text page mapping differs: {len(pages)} != {len(images)}')
        words = {}
        with output_base.with_suffix('.tsv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t', quoting=csv.QUOTE_NONE):
                if row.get('level') == '5' and row.get('text', '').strip():
                    page = int(row['page_num'])
                    if not 1 <= page <= len(images):
                        raise ValueError('TSV page outside image range')
                    words.setdefault(page, []).append(row)
        flags = []
        with (evidence / 'mapping.jsonl').open('w', encoding='utf-8') as mapping:
            for i, (path, text) in enumerate(zip(images, pages), 1):
                lines = [line for line in text.strip('\r\n').splitlines() if line.strip()]
                page_words = words.get(i, [])
                score = min((max(0.0, float(row['conf']))/100 for row in page_words), default=0.0)
                result = {'input_path': str(prepared_paths[i-1]), 'rec_texts': lines,
                          'rec_scores': [score] * len(lines), 'rec_polys': [], 'rec_boxes': [],
                          'engine': 'tesseract', 'word_confidences': page_words}
                result_name = path.stem + '_res.json'
                write_json(evidence / 'results' / result_name, result)
                write_json(output / result_name, result)
                item = {'index': i-1, 'basename': path.stem, 'source': str(path),
                        'original': str(evidence / 'original' / path.name),
                        'original_sha256': digest_file(evidence / 'original' / path.name),
                        'prepared': str(prepared_paths[i-1]),
                        'result': str(evidence / 'results' / result_name)}
                mapping.write(json.dumps(item, ensure_ascii=False)+'\n')
                if not lines or score < .90:
                    flags.append({'index': i-1, 'basename': path.stem, 'texts': lines,
                                  'scores': result['rec_scores'], 'original': item['original'],
                                  'result': item['result']})
        write_json(evidence / 'flags.json', flags)
        write_json(track / 'paddle-flags.json', {'run_id': run_id, 'flags': flags})
        state.update(completed=len(images), flagged=len(flags))
        checkpoint('completed')
        print(json.dumps({'pgs_ocr_progress': state}, ensure_ascii=False), flush=True)
    except Exception as exc:
        state['error'] = str(exc)
        checkpoint('failed')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
