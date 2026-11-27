"""PGS image/timing validation helpers. No OCR, publication, or MD mutations."""

from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET


from pgs_config import TMP_ROOT

DEFAULT_FONT = Path(os.environ.get("PGS_REVIEW_FONT", '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')).expanduser()
TIME = r'-?\d+:[0-5]\d:[0-5]\d[,.]\d{3}'
HEADER = re.compile(
    r'(?m)^[ \t]*(?P<index>\d+)[ \t]*\n[ \t]*(?P<start>' + TIME +
    r')[ \t]+-->[ \t]+(?P<end>' + TIME + r')(?:[ \t]+[^\n]*)?(?:\n|$)'
)


class ValidationError(ValueError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def _milliseconds(value):
    sign = -1 if value.startswith('-') else 1
    hours, minutes, seconds, milliseconds = map(int, re.split(r'[:,.]', value.lstrip('-')))
    return sign * (((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds)


def parse_srt(path):
    """Parse UTF-8/BOM SRT, including empty-text seconv timing baselines."""
    text = Path(path).read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    matches = list(HEADER.finditer(text))
    if not matches:
        if text.strip():
            raise ValidationError(f'SRT 中没有合法时间码条目：{path}')
        return []
    if text[:matches[0].start()].strip():
        raise ValidationError(f'SRT 首条之前存在无法解析的内容：{path}')
    result = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip('\n')
        # A malformed timestamp must not silently become subtitle dialogue.
        if re.search(r'(?m)^\s*-?\d+:\d+:\d+[,.]\d+\s+-->', body):
            raise ValidationError(f'SRT 第 {match["index"]} 项之后存在无法解析的时间码：{path}')
        result.append({
            'index': int(match['index']),
            'start_ms': _milliseconds(match['start']),
            'end_ms': _milliseconds(match['end']),
            'text': body,
        })
    return result


def _entries(baseline):
    return parse_srt(baseline) if isinstance(baseline, (str, os.PathLike)) else list(baseline)


def _local_name(element):
    return element.tag.rsplit('}', 1)[-1]


def _bdn_time(value, fps):
    match = re.fullmatch(r'(\d+):([0-5]\d):([0-5]\d):(\d+)', value or '')
    if not match:
        raise ValidationError(f'无法解析 BDN 帧时间码：{value}')
    hours, minutes, seconds, frames = map(int, match.groups())
    if frames >= math.ceil(fps):
        raise ValidationError(f'BDN 帧编号超出帧率：{value}，fps={fps}')
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + Fraction(frames * 1000, 1) / fps


def read_bdn(xml, baseline):
    """Map ONE Graphic per BDN event to the exact SRT baseline milliseconds.

    Checks event counts, sequential baseline indexes, bitmap dimensions, and both
    BDN frame-time endpoints within one frame of the independent SRT baseline.
    BDN's generic Language=eng metadata is deliberately not a language source.
    """
    from PIL import Image

    xml = Path(xml).resolve()
    timing = _entries(baseline)
    root = ET.parse(xml).getroot()
    formats = [el for el in root.iter() if _local_name(el) == 'Format']
    if len(formats) != 1:
        raise ValidationError('BDN 必须有且仅有一个 Format 元素')
    fmt = formats[0]
    if fmt.get('DropFrame', 'false').lower() == 'true':
        raise ValidationError('当前映射器不接受 DropFrame=true；需先明确其 SMPTE 时间算法')
    try:
        fps = Fraction(fmt.get('FrameRate', ''))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValidationError('BDN 帧率不合法') from exc
    if fps <= 0 or fps > 240:
        raise ValidationError(f'BDN 帧率不合理：{fps}')
    nodes = [el for el in root.iter() if _local_name(el) == 'Event']
    declared = [el.get('NumberofEvents') for el in root.iter()
                if _local_name(el) == 'Events' and el.get('NumberofEvents') is not None]
    if len(declared) != 1 or int(declared[0]) != len(nodes) or len(nodes) != len(timing):
        raise ValidationError('BDN 声明数、实际事件数与时间码基线数量不一致', {
            'declared': declared, 'bdn_count': len(nodes), 'baseline_count': len(timing),
        })
    if not nodes:
        raise ValidationError('BDN 与基线均无事件，不能当作完整字幕轨')
    frame_ms = Fraction(1000, 1) / fps
    result = []
    for position, (node, base) in enumerate(zip(nodes, timing), 1):
        if base['index'] != position:
            raise ValidationError(f'基线序号不连续：位置 {position}，序号 {base["index"]}')
        if base['start_ms'] < 0 or base['end_ms'] <= base['start_ms']:
            raise ValidationError(f'基线存在非法起止时间：{position}')
        if position > 1 and base['start_ms'] < timing[position - 2]['start_ms']:
            raise ValidationError(f'基线开始时间逆序：{position}')
        graphics = [el for el in node if _local_name(el) == 'Graphic']
        if len(graphics) != 1:
            raise ValidationError(f'BDN 第 {position} 个事件含 {len(graphics)} 个 Graphic，不能单图映射')
        graphic = graphics[0]
        image_path = (xml.parent / (graphic.text or '').strip()).resolve()
        if not image_path.is_relative_to(xml.parent) or not image_path.is_file():
            raise ValidationError(f'BDN 图片缺失或逃逸图片目录：{position}，{image_path}')
        width, height = int(graphic.get('Width', '0')), int(graphic.get('Height', '0'))
        if width <= 0 or height <= 0:
            raise ValidationError(f'BDN 图片声明尺寸不合法：{position}')
        with Image.open(image_path) as picture:
            if picture.size != (width, height):
                raise ValidationError(f'BDN 图片尺寸与 Graphic 声明不一致：{position}')
        bdn_start, bdn_end = _bdn_time(node.get('InTC'), fps), _bdn_time(node.get('OutTC'), fps)
        start_delta, end_delta = bdn_start - base['start_ms'], bdn_end - base['end_ms']
        if abs(start_delta) > frame_ms or abs(end_delta) > frame_ms:
            raise ValidationError(f'BDN 第 {position} 项与基线偏差超过一帧', {
                'index': position, 'frame_ms': float(frame_ms),
                'start_delta_ms': float(start_delta), 'end_delta_ms': float(end_delta),
            })
        # Do not reject BDN zero-length rounded events: a valid short baseline
        # event can quantize to equal BDN InTC/OutTC and must still be retained.
        if bdn_end < bdn_start:
            raise ValidationError(f'BDN 第 {position} 项帧时间逆序')
        result.append({
            **base, 'image_path': str(image_path), 'width': width, 'height': height,
            'x': int(graphic.get('X', '0')), 'y': int(graphic.get('Y', '0')),
            'frame_rate': float(fps), 'bdn_start_ms': float(bdn_start),
            'bdn_end_ms': float(bdn_end), 'start_delta_ms': float(start_delta),
            'end_delta_ms': float(end_delta),
        })
    return result


def _visible_text(text):
    return re.sub(r'</?(?:i|b|u|font)(?:\s+[^>]*)?>', '', text, flags=re.I).strip(
        ' \t\r\n\u200b\u200c\u200d\ufeff')


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _reviewed_nontext(actual, expected, records, source_sup, bdn):
    """Verify explicit image-review exceptions; never infer them from OCR failure."""
    accepted, errors = {}, []
    if records is None:
        return accepted, errors
    if not isinstance(records, list):
        return accepted, [{'kind': 'nontext_records_type', 'detail': 'reviewed_nontext must be a list'}]
    if not records:
        return accepted, errors
    if source_sup is None or bdn is None:
        return accepted, [{'kind': 'nontext_context', 'detail': 'actual source_sup and BDN XML are required'}]
    try:
        source_hash = _file_sha256(source_sup)
        events = read_bdn(bdn, expected)
    except (OSError, ValueError, ET.ParseError) as exc:
        return accepted, [{'kind': 'nontext_context', 'detail': str(exc)}]
    seen = set()
    for record_position, record in enumerate(records, 1):
        if not isinstance(record, dict):
            errors.append({'kind': 'nontext_record_type', 'record': record_position})
            continue
        index = record.get('index')
        if type(index) is not int or not 1 <= index <= len(expected) or index > len(actual):
            errors.append({'kind': 'nontext_index', 'record': record_position, 'index': index})
            continue
        if index in seen:
            errors.append({'kind': 'nontext_duplicate', 'index': index})
            continue
        seen.add(index)
        before = len(errors)
        base, row, event = expected[index - 1], actual[index - 1], events[index - 1]
        if record.get('confirmed') is not True:
            errors.append({'kind': 'nontext_unconfirmed', 'index': index})
        reason = record.get('reason')
        if not isinstance(reason, str) or not reason.strip() or not any(ch.isalnum() for ch in reason):
            errors.append({'kind': 'nontext_reason', 'index': index})
        if (type(record.get('start_ms')) is not int or type(record.get('end_ms')) is not int
                or (record['start_ms'], record['end_ms']) != (base['start_ms'], base['end_ms'])
                or row['index'] != index
                or (row['start_ms'], row['end_ms']) != (base['start_ms'], base['end_ms'])):
            errors.append({'kind': 'nontext_timing', 'index': index})
        try:
            image_hash = _file_sha256(event['image_path'])
        except OSError as exc:
            errors.append({'kind': 'nontext_image_context', 'index': index, 'detail': str(exc)})
            continue
        for field, current, kind in (
                ('sup_sha256', source_hash, 'nontext_source_hash'),
                ('image_sha256', image_hash, 'nontext_image_hash')):
            supplied = record.get(field)
            if (not isinstance(supplied, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', supplied)
                    or supplied.lower() != current):
                errors.append({'kind': kind, 'index': index})
        # Preserve a literally empty raw event; do not hide text or fabricated
        # substitutes behind a nontext label (even formatting-only markup).
        if row['text'].strip(' \t\r\n\u200b\u200c\u200d\ufeff'):
            errors.append({'kind': 'nontext_has_text', 'index': index})
        if len(errors) == before:
            accepted[index] = dict(record)
    return accepted, errors


def publishable_entries(entries, *, reviewed_nontext=None, source_sup=None, bdn=None, baseline=None):
    """Return text events, omitting only hash-bound, confirmed nontext events.

    Validate the full unmerged SRT first. This helper rechecks each exception's
    source/image/time binding before filtering; it never removes an unannotated
    empty cue. Original event indexes and exact times remain in returned rows,
    so merge_adjacent/write_srt can retain source_indices while numbering output.
    """
    actual = _entries(entries)
    if reviewed_nontext is None or reviewed_nontext == []:
        return [dict(row) for row in actual]
    if baseline is None:
        raise ValidationError('使用无文字例外生成成品时必须提供完整时间码基线',
                              {'errors': [{'kind': 'nontext_context', 'detail': 'baseline required'}]})
    accepted, errors = _reviewed_nontext(actual, _entries(baseline), reviewed_nontext, source_sup, bdn)
    if errors:
        raise ValidationError('无文字事件标注未通过绑定校验', {'errors': errors})
    return [dict(row) for row in actual if row['index'] not in accepted]


def validate_ocr_srt(path, baseline, duration, *, end_tolerance_ms=2000, raise_on_error=True,
                     reviewed_nontext=None, source_sup=None, bdn=None):
    """Validate UNMERGED OCR output; duration is media duration in SECONDS.

    All exact baseline times and even sub-250ms events must be present. Overlaps
    are allowed; descending start times, empty captions, and out-of-media timing
    fail. A 2-second end margin handles container duration rounding, never clips.
    The only empty-event exception is an explicit confirmed reviewed_nontext
    record bound to the exact index/times and actual SUP/BDN PNG hashes. The raw
    SRT still retains every event. No records preserves the original behavior.
    """
    actual, expected = parse_srt(path), _entries(baseline)
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration 必须是以秒为单位的有限正数')
    duration_ms = round(duration * 1000)
    errors, warnings = [], []
    nontext, annotation_errors = _reviewed_nontext(actual, expected, reviewed_nontext, source_sup, bdn)
    errors.extend(annotation_errors)
    text_indices, unresolved_empty = [], []
    if not expected:
        errors.append({'kind': 'empty_baseline'})
    if len(actual) != len(expected):
        errors.append({'kind': 'count', 'expected': len(expected), 'actual': len(actual)})
    for position, row in enumerate(actual, 1):
        if row['index'] != position:
            errors.append({'kind': 'index', 'position': position, 'actual': row['index']})
        start, end, text = row['start_ms'], row['end_ms'], row['text']
        if start < 0 or end <= start:
            errors.append({'kind': 'invalid_interval', 'index': position, 'start_ms': start, 'end_ms': end})
        if position > 1 and start < actual[position - 2]['start_ms']:
            errors.append({'kind': 'descending_start', 'index': position})
        if start > duration_ms + end_tolerance_ms or end > duration_ms + end_tolerance_ms:
            errors.append({'kind': 'outside_duration', 'index': position, 'end_ms': end, 'duration_ms': duration_ms})
        if not _visible_text(text):
            if position not in nontext:
                errors.append({'kind': 'empty_text', 'index': position})
                unresolved_empty.append(position)
        elif any(not line.strip() for line in text.splitlines()):
            warnings.append({'kind': 'blank_text_line', 'index': position})
        if _visible_text(text):
            text_indices.append(position)
        if position <= len(expected):
            base = expected[position - 1]
            if (start, end) != (base['start_ms'], base['end_ms']):
                errors.append({'kind': 'timing_mismatch', 'index': position,
                               'expected': [base['start_ms'], base['end_ms']], 'actual': [start, end]})
    short_expected = [row['index'] for row in expected if 0 < row['end_ms'] - row['start_ms'] < 250]
    report = {
        'ok': not errors, 'path': str(path), 'entries': len(actual),
        'baseline_entries': len(expected), 'short_baseline_entries': short_expected,
        'duration_ms': duration_ms, 'end_tolerance_ms': end_tolerance_ms,
        'errors': errors, 'warnings': warnings,
        'reviewed_nontext_indices': sorted(nontext),
        'coverage': {
            'baseline_events': len(expected), 'raw_events': len(actual),
            'text_events': len(text_indices), 'text_indices': text_indices,
            'reviewed_nontext_events': len(nontext), 'reviewed_nontext_indices': sorted(nontext),
            'unresolved_empty_events': len(unresolved_empty), 'unresolved_empty_indices': unresolved_empty,
            'text_events_required': len(expected) - len(nontext),
            'complete': not errors and len(text_indices) + len(nontext) == len(expected),
        },
    }
    if errors and raise_on_error:
        raise ValidationError(f'OCR SRT 校验失败：{len(errors)} 个问题，首项 {errors[0]}', report)
    return report


def _events(value):
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (str, os.PathLike)):
        return read_bdn(value[0], value[1])
    return list(value)


def _pixels(event):
    from PIL import Image
    with Image.open(event['image_path']) as picture:
        rgba = picture.convert('RGBA')
        dimensions = rgba.size
        digest = hashlib.sha256(struct.pack('>II', *dimensions) + rgba.tobytes()).hexdigest()
    if dimensions != (event['width'], event['height']):
        raise ValidationError(f'读取后图片尺寸发生变化：{event["image_path"]}')
    return dimensions, digest


def track_fingerprint(events):
    """Hash every exact event time, RGBA dimension and decoded pixel hash, excluding X/Y."""
    digest = hashlib.sha256(b'pgs-exact-event-rgba-v1\0')
    events = _events(events)
    digest.update(struct.pack('>Q', len(events)))
    for event in events:
        (width, height), pixel_hash = _pixels(event)
        digest.update(struct.pack('>qqII', event['start_ms'], event['end_ms'], width, height))
        digest.update(bytes.fromhex(pixel_hash))
    return digest.hexdigest()


def compare_tracks(left, right):
    """Return identical=True only after comparing ALL exact times and decoded RGBA pixels.

    Accepts read_bdn event lists or (xml_path, baseline) tuples. Screen placement
    X/Y is ignored. False may return early at the first proven difference.
    """
    left, right = _events(left), _events(right)
    report = {'identical': False, 'left_count': len(left), 'right_count': len(right), 'compared': 0}
    if len(left) != len(right):
        return {**report, 'reason': 'event_count'}
    if not left:
        return {**report, 'reason': 'empty_tracks_are_not_proven_duplicates'}
    for position, (a, b) in enumerate(zip(left, right), 1):
        if (a['start_ms'], a['end_ms']) != (b['start_ms'], b['end_ms']):
            return {**report, 'reason': 'exact_timing', 'first_difference': position}
        if _pixels(a) != _pixels(b):
            return {**report, 'reason': 'rgba_pixels_or_dimensions', 'first_difference': position}
        report['compared'] = position
    return {**report, 'identical': True, 'reason': 'all_exact_times_and_rgba_pixels_equal'}


def _output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(TMP_ROOT):
        raise ValueError(f'校验模块的所有输出必须位于 TMPDIR：{path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _srt_time(milliseconds):
    value = int(milliseconds)
    if value < 0 or value != milliseconds:
        raise ValueError(f'SRT 时间必须为非负整数毫秒：{milliseconds}')
    seconds, ms = divmod(value, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'


def write_srt(path, entries, *, overwrite=False):
    """Write an unmerged or merged SRT in TMPDIR; never overwrites by default."""
    path = _output_path(path)
    blocks = []
    for index, row in enumerate(entries, 1):
        if row['end_ms'] <= row['start_ms']:
            raise ValidationError(f'拒绝写入非正时长条目：{index}')
        text = str(row.get('text', '')).replace('\r\n', '\n').replace('\r', '\n').strip('\n')
        blocks.append(f'{index}\n{_srt_time(row["start_ms"])} --> {_srt_time(row["end_ms"])}\n{text}\n\n')
    with path.open('w' if overwrite else 'x', encoding='utf-8', newline='\n') as output:
        output.write(''.join(blocks))
        output.flush()
        os.fsync(output.fileno())
    return path


def merge_adjacent(entries):
    """Merge only identical nonempty text with touching/overlapping time intervals.

    Returns new dictionaries, leaving original entries untouched. source_indices
    records every original cue that contributed to each merged cue.
    """
    result = []
    previous_start = -1
    for position, source in enumerate(entries, 1):
        row = dict(source)
        start, end = row['start_ms'], row['end_ms']
        if start < 0 or end <= start or start < previous_start:
            raise ValidationError(f'合并输入的时间顺序或时长不合法：{position}')
        previous_start = start
        row['source_indices'] = list(source.get('source_indices', [source.get('index', position)]))
        if (result and str(row.get('text', '')).strip() and row.get('text') == result[-1].get('text')
                and start <= result[-1]['end_ms']):
            result[-1]['end_ms'] = max(result[-1]['end_ms'], end)
            result[-1]['source_indices'].extend(row['source_indices'])
        else:
            row['index'] = len(result) + 1
            result.append(row)
    return result


def sample_montage(events, output, *, indices=None, low_score_indices=(), texts=None,
                   max_items=20, max_width=1600, font_path=DEFAULT_FONT):
    """Create a numbered, color-preserving review PNG in TMPDIR.

    All indices are ONE-BASED SRT indices. Defaults sample beginning, quarter,
    middle, three-quarter, end; requested low-score samples follow. Returned
    omitted_indices makes any max_items truncation explicit. texts is either a
    one-based mapping or an event-ordered text/dict sequence for OCR comparison.
    """
    from PIL import Image, ImageDraw, ImageFont
    events = _events(events)
    if not events or max_items < 1 or max_width < 200:
        raise ValueError('蒙太奇需要事件，max_items>=1 且 max_width>=200')
    requested = list(indices) if indices is not None else [round((len(events) - 1) * f) + 1 for f in (0, .25, .5, .75, 1)]
    requested.extend(low_score_indices)
    selected = list(dict.fromkeys(int(i) for i in requested))
    if any(i < 1 or i > len(events) for i in selected):
        raise ValueError('蒙太奇条目序号超出范围')
    omitted, selected = selected[max_items:], selected[:max_items]
    path = _output_path(output)
    if path.exists():
        raise FileExistsError(path)
    font = ImageFont.truetype(str(font_path), 20)
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    rows = []
    for index in selected:
        event = events[index - 1]
        with Image.open(event['image_path']) as original:
            rgba = original.convert('RGBA')
            picture = Image.new('RGBA', rgba.size, (32, 32, 32, 255))
            picture.alpha_composite(rgba)
            picture = picture.convert('RGB')
        if picture.width > max_width - 24:
            ratio = (max_width - 24) / picture.width
            picture = picture.resize((max_width - 24, max(1, round(picture.height * ratio))), Image.Resampling.LANCZOS)
        labels = [f'#{index}  {_srt_time(event["start_ms"])} --> {_srt_time(event["end_ms"])}']
        if texts is not None:
            value = texts.get(index, '') if isinstance(texts, dict) else texts[index - 1]
            if isinstance(value, dict):
                value = value.get('text', '')
            for line in str(value).splitlines() or ['']:
                current = ''
                for char in line:
                    if current and probe.textlength(current + char, font=font) > max_width - 24:
                        labels.append(current)
                        current = char
                    else:
                        current += char
                labels.append(current)
        row = Image.new('RGB', (max_width, 16 + len(labels) * 28 + picture.height + 16), (24, 24, 24))
        draw = ImageDraw.Draw(row)
        for number, label in enumerate(labels):
            draw.text((12, 8 + number * 28), label, font=font, fill=(235, 235, 235) if number == 0 else (250, 218, 115))
        row.paste(picture, (12, 16 + len(labels) * 28))
        rows.append(row)
    montage = Image.new('RGB', (max_width, sum(row.height for row in rows) + 8 * (len(rows) - 1)), (95, 95, 95))
    y = 0
    for row in rows:
        montage.paste(row, (0, y))
        y += row.height + 8
    with path.open('xb') as destination:
        montage.save(destination, format='PNG')
    return {'path': str(path), 'selected_indices': selected, 'omitted_indices': omitted,
            'width': montage.width, 'height': montage.height}
