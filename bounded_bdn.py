"""Bounded, byte-preserving PGS -> BDN export with exact timing evidence.

Only a cleared display followed by a later epoch-start is a cut point. A SUP
which cannot be split within the hard decoded-image budget is rejected, never
passed to a whole-track fallback. No original media, SUP or baseline is edited.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import uuid
import xml.etree.ElementTree as ET

from validation import parse_srt, read_bdn
from bdn_timing_alignment import align_batch
from pgs_config import SECONV, TMP_ROOT


TARGET_RGBA_BYTES = 64 * 1024 ** 2
MAX_RGBA_BYTES = 128 * 1024 ** 2
SCHEMA = 'bounded-bdn-v1'


class BoundedExportError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise BoundedExportError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def _name(element):
    return element.tag.rsplit('}', 1)[-1]


def _one(root, name, predicate=lambda element: True):
    nodes = [element for element in root.iter()
             if _name(element) == name and predicate(element)]
    _require(len(nodes) == 1, f'BDN must contain one {name}: found {len(nodes)}')
    return nodes[0]


def inspect_sup(sup, *, target_rgba_bytes=TARGET_RGBA_BYTES,
                max_rgba_bytes=MAX_RGBA_BYTES):
    """Validate PGS references and return complete, independently closed batches.

    The estimate charges every display set for the larger of all new decoded
    objects or its composed image/referenced objects. It is a decoded RGBA
    budget, not a promise about the converter's total process RSS.
    """
    _require(0 < target_rgba_bytes <= max_rgba_bytes <= MAX_RGBA_BYTES,
             'Invalid bounded-export image budget')
    sup = Path(sup)
    display_sets = []
    objects, palettes, windows, unfinished = {}, set(), {}, {}
    current = None
    video = None
    epoch_count = 0
    epoch_pts = None
    with sup.open('rb') as source:
        while True:
            offset = source.tell()
            header = source.read(13)
            if not header:
                break
            _require(len(header) == 13, f'Truncated SUP header at {offset}')
            magic, pts, dts, kind, size = struct.unpack('>2sIIBH', header)
            _require(magic == b'PG', f'Invalid SUP magic at {offset}')
            payload = source.read(size)
            _require(len(payload) == size, f'Truncated SUP segment at {offset}')
            _require(kind in (0x14, 0x15, 0x16, 0x17, 0x80),
                     f'Unsupported SUP segment {kind:#x} at {offset}')
            if kind == 0x16:
                _require(current is None, f'Display set lacks END before {offset}')
                _require(size >= 11, f'Truncated PCS at {offset}')
                width, height = struct.unpack_from('>HH', payload)
                state, palette_id, count = payload[7], payload[9], payload[10]
                _require(width > 0 and height > 0 and state in (0, 0x40, 0x80),
                         f'Invalid PCS state/dimensions at {offset}')
                _require(video in (None, (width, height)),
                         'Changing PGS video dimensions require separate review')
                video = (width, height)
                _require(not display_sets or pts >= display_sets[-1]['pts'],
                         'Non-monotonic PCS timestamps/wraparound are unsupported')
                if state == 0x80:
                    _require(not unfinished, f'Incomplete ODS before epoch at {offset}')
                    objects, palettes, windows = {}, set(), {}
                    epoch_count += 1
                    epoch_pts = pts
                _require(epoch_count, 'SUP must begin at an independently decodable epoch')
                references, cursor = [], 11
                for _ in range(count):
                    _require(cursor + 8 <= size, f'Truncated PCS object at {offset}')
                    object_id, window_id, flags, x, y = struct.unpack_from('>HBBHH', payload, cursor)
                    cursor += 8
                    _require(flags & 0x3f == 0, f'Unsupported PCS object flags at {offset}')
                    crop = None
                    if flags & 0x80:
                        _require(cursor + 8 <= size, f'Truncated PCS crop at {offset}')
                        crop = struct.unpack_from('>HHHH', payload, cursor)
                        cursor += 8
                    references.append((object_id, window_id, x, y, crop))
                _require(cursor == size, f'Unexpected PCS trailing bytes at {offset}')
                current = {'start': offset, 'pts': pts, 'epoch_pts': epoch_pts, 'state': state,
                           'objects': count, 'new_rgba': 0, 'palette': palette_id,
                           'references': references}
                continue
            _require(current is not None, f'Segment outside a display set at {offset}')
            if kind == 0x14:
                _require(size >= 2 and (size - 2) % 5 == 0, f'Invalid PDS at {offset}')
                palettes.add(payload[0])
            elif kind == 0x17:
                _require(size >= 1 and size == 1 + 9 * payload[0], f'Invalid WDS at {offset}')
                for cursor in range(1, size, 9):
                    window_id, x, y, width, height = struct.unpack_from('>BHHHH', payload, cursor)
                    _require(width > 0 and height > 0 and x + width <= video[0]
                             and y + height <= video[1], f'Invalid WDS rectangle at {offset}')
                    windows[window_id] = (x, y, width, height)
            elif kind == 0x15:
                _require(size >= 4, f'Truncated ODS at {offset}')
                object_id, version, flags = struct.unpack_from('>HBB', payload)
                _require(flags & 0x3f == 0, f'Unsupported ODS flags at {offset}')
                if flags & 0x80:
                    _require(size >= 11 and object_id not in unfinished,
                             f'Invalid ODS first fragment at {offset}')
                    length = int.from_bytes(payload[4:7], 'big')
                    width, height = struct.unpack_from('>HH', payload, 7)
                    _require(length >= 4 and width > 0 and height > 0
                             and width <= video[0] and height <= video[1],
                             f'Invalid ODS dimensions/length at {offset}')
                    unfinished[object_id] = {'version': version, 'remaining': length - (size - 7),
                                             'width': width, 'height': height}
                    current['new_rgba'] += width * height * 4
                else:
                    _require(object_id in unfinished, f'ODS continuation without first at {offset}')
                    _require(unfinished[object_id]['version'] == version,
                             f'ODS version changes during fragments at {offset}')
                    unfinished[object_id]['remaining'] -= size - 4
                obj = unfinished[object_id]
                _require(obj['remaining'] >= 0, f'ODS declared length exceeded at {offset}')
                if flags & 0x40:
                    _require(obj['remaining'] == 0, f'Incomplete ODS last fragment at {offset}')
                    objects[object_id] = (obj['width'], obj['height'])
                    del unfinished[object_id]
                else:
                    _require(obj['remaining'] > 0, f'ODS missing last-fragment flag at {offset}')
            else:
                _require(size == 0 and not unfinished, f'END with incomplete ODS at {offset}')
                rectangles, referenced_rgba = [], 0
                for object_id, window_id, x, y, crop in current['references']:
                    _require(object_id in objects and window_id in windows
                             and current['palette'] in palettes,
                             f'Unresolved object/window/palette reference at {current["start"]}')
                    width, height = objects[object_id]
                    referenced_rgba += width * height * 4
                    if crop:
                        cx, cy, cw, ch = crop
                        _require(cw > 0 and ch > 0 and cx + cw <= width and cy + ch <= height,
                                 f'Invalid object crop at {current["start"]}')
                        width, height = cw, ch
                    _require(x + width <= video[0] and y + height <= video[1],
                             f'Object outside video canvas at {current["start"]}')
                    rectangles.append((x, y, x + width, y + height))
                composed_rgba = 0
                if rectangles:
                    composed_rgba = ((max(rect[2] for rect in rectangles) - min(rect[0] for rect in rectangles))
                                     * (max(rect[3] for rect in rectangles) - min(rect[1] for rect in rectangles)) * 4)
                current['rgba_bytes'] = max(current.pop('new_rgba'), referenced_rgba, composed_rgba)
                current['end'] = source.tell()
                del current['references'], current['palette']
                display_sets.append(current)
                current = None
        _require(current is None and not unfinished, 'SUP ends inside an incomplete display set')
        source_size = source.tell()
    _require(display_sets and display_sets[0]['state'] == 0x80, 'SUP has no initial epoch')
    _require(display_sets[-1]['objects'] == 0,
             'SUP ends with an uncleared display; refuse to invent end-of-file timing')
    groups, start = [], 0
    for index in range(1, len(display_sets) + 1):
        previous = display_sets[index - 1]
        at_end = index == len(display_sets)
        safe = at_end or (previous['objects'] == 0 and display_sets[index]['state'] == 0x80
                          and display_sets[index]['pts'] > previous['epoch_pts'])
        if not safe:
            continue
        members = display_sets[start:index]
        cost = sum(member['rgba_bytes'] for member in members)
        _require(cost <= max_rgba_bytes,
                 f'No safe clear/epoch cut within {max_rgba_bytes // 1024**2} MiB: '
                 f'closed group at byte {members[0]["start"]} needs {cost / 1024**2:.2f} MiB')
        groups.append({'start': members[0]['start'], 'end': previous['end'],
                       'first_pts': members[0]['pts'], 'last_pts': previous['pts'],
                       'rgba_bytes': cost, 'display_sets': len(members),
                       'epochs': sum(member['state'] == 0x80 for member in members)})
        start = index
    batches, current = [], None
    for group in groups:
        if current is not None and current['rgba_bytes'] + group['rgba_bytes'] > max_rgba_bytes:
            batches.append(current)
            current = None
        if current is None:
            current = dict(group)
        else:
            current.update(end=group['end'], last_pts=group['last_pts'])
            for key in ('rgba_bytes', 'display_sets', 'epochs'):
                current[key] += group[key]
        if current['rgba_bytes'] >= target_rgba_bytes:
            batches.append(current)
            current = None
    if current is not None:
        # Avoid a final empty fragment when the previous fragment can own it.
        if batches and current['rgba_bytes'] == 0:
            batches[-1].update(end=current['end'], last_pts=current['last_pts'])
            for key in ('display_sets', 'epochs'):
                batches[-1][key] += current[key]
        else:
            batches.append(current)
    _require(batches[0]['start'] == 0 and batches[-1]['end'] == source_size,
             'Batch plan does not cover the complete source')
    _require(all(left['end'] == right['start'] for left, right in zip(batches, batches[1:])),
             'Batch plan has missing or duplicated source bytes')
    return {'source_size': source_size, 'video_dimensions': list(video),
            'display_sets': len(display_sets), 'epochs': epoch_count,
            'closed_groups': len(groups), 'target_rgba_bytes': target_rgba_bytes,
            'max_rgba_bytes': max_rgba_bytes, 'batches': batches}


def _copy_range(sup, fragment, start, end):
    digest = hashlib.sha256()
    with Path(sup).open('rb') as source, Path(fragment).open('xb') as destination:
        source.seek(start)
        remaining = end - start
        while remaining:
            block = source.read(min(remaining, 1024 * 1024))
            _require(block, 'SUP changed or truncated while copying a batch')
            destination.write(block)
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _convert(sup, mode, directory, expected, run_command):
    directory.mkdir(parents=True, exist_ok=True)
    command = [SECONV, sup, mode, f'--output-folder:{directory}', '--json']
    if mode == 'subrip':
        command.append('--time-codes-only')
    result_file = directory / 'seconv-result.json'
    run_command(command, result_file, stderr_path=directory / 'seconv.stderr.log')
    result = json.loads(result_file.read_text(encoding='utf-8-sig'))
    _require(result.get('success') and not result.get('failedFiles') and expected.is_file(),
             f'Incomplete seconv {mode} export: {result_file}')


def _times(entries):
    return [(entry['start_ms'], entry['end_ms']) for entry in entries]


def export_bdn(sup, output_dir, expected_xml, run_command, baseline, *,
               target_rgba_bytes=TARGET_RGBA_BYTES, max_rgba_bytes=MAX_RGBA_BYTES):
    """Return expected_xml only after a fully verified, atomic BDN commit.

    run_command uses the queue's existing process-group/signal handling. Each
    batch proves exact millisecond equality against the full independent
    baseline. An ordinary over-split batch additionally needs identical raw
    pixels/properties and source PCS proof before exact partition alignment.
    read_bdn checks seconv's frame-quantized BDN representation throughout.
    Retained .bounded-bdn-* directories contain inputs, logs and audit.json.
    """
    sup, output_dir, expected_xml, baseline = map(Path, (sup, output_dir, expected_xml, baseline))
    _require(TMP_ROOT != Path('/tmp') and not TMP_ROOT.is_relative_to('/tmp'),
             'TMPDIR must not be /tmp')
    for path in (output_dir, expected_xml):
        _require(path.resolve().is_relative_to(TMP_ROOT),
                 f'BDN intermediates must remain under TMPDIR: {path}')
    _require(expected_xml.parent.parent.resolve() == output_dir.resolve(),
             'Expected BDN path must be output_dir / track-name / index.xml')
    if expected_xml.exists():
        read_bdn(expected_xml, baseline)
        return expected_xml
    if expected_xml.parent.exists():
        _require(expected_xml.parent.is_dir() and not any(expected_xml.parent.iterdir()),
                 f'Refuse an unknown nonempty BDN directory: {expected_xml.parent}')
    full_timing = parse_srt(baseline)
    _require(full_timing and all(item['index'] == index for index, item in enumerate(full_timing, 1)),
             'Full timing baseline must have nonempty sequential entries')
    original_snapshot, baseline_snapshot = _snapshot(sup), _snapshot(baseline)
    original_hash, baseline_hash = _sha256(sup), _sha256(baseline)
    plan = inspect_sup(sup, target_rgba_bytes=target_rgba_bytes, max_rgba_bytes=max_rgba_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = output_dir / f'.bounded-bdn-{token}'
    staging.mkdir()
    merged = staging / 'merged'
    merged.mkdir()
    audit_path = staging / 'audit.json'
    audit = {'schema': SCHEMA, 'status': 'running', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'source': {'path': str(sup), **original_snapshot, 'sha256': original_hash},
             'baseline': {'path': str(baseline), **baseline_snapshot, 'sha256': baseline_hash,
                          'entries': len(full_timing)},
             'source_baseline': {'path': str(baseline), **baseline_snapshot, 'sha256': baseline_hash,
                                 'entries': len(full_timing)},
             'event_count_semantics': {
                 'source_event_count': 'legacy alias of raw_export_event_count; summed raw chunk timing rows, not physical PGS display sets',
                 'raw_export_event_count': 'summed original fragment time-codes-only rows before alignment/filtering',
                 'canonical_count': 'retained events matching the complete verification baseline',
                 'source_baseline_event_count': 'events in the original complete time-codes-only baseline'},
             'cut_policy': 'complete clear END, then strictly later epoch_start; preserve every source byte',
             'plan': plan, 'batches': [], 'expected_xml': str(expected_xml)}
    _json(audit_path, audit)
    cursor, source_cursor, merged_root, event_container, format_attributes = 0, 0, None, None, None
    all_times, event_signatures = [], []
    source_times = []
    try:
        for number, batch in enumerate(plan['batches'], 1):
            directory = staging / f'batch-{number:04d}'
            directory.mkdir()
            fragment = directory / f'batch-{number:04d}.sup'
            fragment_hash = _copy_range(sup, fragment, batch['start'], batch['end'])
            timing_path = directory / 'timing' / f'{fragment.stem}.srt'
            _convert(fragment, 'subrip', timing_path.parent, timing_path, run_command)
            timing = parse_srt(timing_path)
            _require(all(item['index'] == index for index, item in enumerate(timing, 1)),
                     f'Batch {number} timing indexes are not sequential')
            expected_timing = full_timing[cursor:cursor + len(timing)]
            raw_exact_match = _times(timing) == _times(expected_timing)
            record = {'number': number, **batch, 'source_sha256': fragment_hash,
                      'baseline_first_index': cursor + 1, 'source_first_index': source_cursor + 1,
                      'source_entries': len(timing), 'entries': len(timing),
                      'raw_export_first_index': source_cursor + 1, 'raw_export_entries': len(timing),
                      'original_chunk_timing': [dict(item) for item in timing],
                      'timing_sha256': _sha256(timing_path),
                      'raw_exact_timing_match': raw_exact_match,
                      'exact_timing_match': raw_exact_match}
            audit['batches'].append(record)
            _json(audit_path, audit)
            source_count = len(timing)
            consumed_count = source_count
            source_times.extend(_times(timing))
            if not timing:
                _require(batch['rgba_bytes'] == 0,
                         f'Batch {number} contains images but seconv returned an empty timing baseline')
                record.update(canonical_consumed_count=0, canonical_count=0)
                _json(audit_path, audit)
                continue
            xml = directory / 'images' / fragment.stem / 'index.xml'
            _convert(fragment, 'bdnxml', xml.parent.parent, xml, run_command)
            record.update(raw_xml=str(xml), raw_xml_sha256=_sha256(xml))
            parsed_events = None
            if not raw_exact_match:
                # Preserve the original BDN/PNG export before any repair. The
                # adapter fails closed unless every raw cue has an exact,
                # pixel/property-certified mapping to the canonical prefix.
                aligned = align_batch(
                    xml, timing, full_timing[cursor:],
                    output_dir=directory / 'aligned',
                    evidence_dir=output_dir.parent / 'alignment-evidence' / token / directory.name,
                    source_sup=fragment, expected_source_sup_sha256=fragment_hash)
                xml, timing = Path(aligned['xml']), aligned['timing']
                parsed_events = aligned['entries']
                consumed_count = aligned['consumed_count']
                _require(consumed_count == len(timing),
                         'Aligned canonical count differs from retained timing count')
                record.update(entries=len(timing), exact_timing_match=True,
                              aggregated_raw_export_entries=source_count - consumed_count,
                              alignment_evidence=str(aligned['evidence']),
                              alignment_evidence_sha256=_sha256(aligned['evidence']))
            _require(_times(timing) == _times(full_timing[cursor:cursor + len(timing)]),
                     'Retained batch differs from the complete baseline')
            if parsed_events is None:
                parsed_events = read_bdn(xml, timing)
            root = ET.parse(xml).getroot()
            fmt = _one(root, 'Format')
            if merged_root is None:
                merged_root = deepcopy(root)
                event_container = _one(merged_root, 'Events', lambda node: node.get('NumberofEvents') is None)
                event_container.clear()
                format_attributes = dict(fmt.attrib)
            _require(dict(fmt.attrib) == format_attributes,
                     f'Batch {number} BDN format/frame rate differs; cannot safely merge')
            nodes = [node for node in root.iter() if _name(node) == 'Event']
            for local_index, (node, parsed) in enumerate(zip(nodes, parsed_events), 1):
                copied = deepcopy(node)
                graphic = _one(copied, 'Graphic')
                source_image = Path(parsed['image_path'])
                image_name = f'bounded-{token}-{cursor + local_index:06d}.png'
                # A hard link retains the exact exported bytes without a second PNG copy.
                os.link(source_image, merged / image_name)
                png_hash = _sha256(source_image)
                graphic.text = image_name
                signature = {'event': dict(copied.attrib), 'graphic': dict(graphic.attrib),
                             'image': image_name, 'png_sha256': png_hash}
                event_signatures.append(signature)
                event_container.append(copied)
            all_times.extend(_times(timing))
            cursor += consumed_count
            source_cursor += source_count
            record.update(xml_sha256=_sha256(xml), frame_format=dict(fmt.attrib),
                          bdn_events=len(nodes), last_global_index=cursor,
                          canonical_consumed_count=consumed_count,
                          canonical_count=consumed_count, source_last_index=source_cursor,
                          raw_export_last_index=source_cursor)
            _json(audit_path, audit)
            if number == 1 or number % 5 == 0 or number == len(plan['batches']):
                print(f'{sup.name}: 分批图片导出 {number}/{len(plan["batches"])} 批，'
                      f'{cursor}/{len(full_timing)} 项已通过精确时间核对', flush=True)
        _require(source_cursor == len(source_times) == sum(item['source_entries'] for item in audit['batches']),
                 'Raw source event counts do not match the retained chunk audit')
        _require(cursor == len(full_timing) and all_times == _times(full_timing),
                 'Merged batches do not exactly cover every full-baseline event in order')
        summary = _one(merged_root, 'Events', lambda node: node.get('NumberofEvents') is not None)
        summary.set('NumberofEvents', str(cursor))
        summary.set('FirstEventInTC', event_container[0].get('InTC'))
        summary.set('LastEventOutTC', event_container[-1].get('OutTC'))
        merged_xml = merged / expected_xml.name
        ET.ElementTree(merged_root).write(merged_xml, encoding='utf-8', xml_declaration=True)
        read_bdn(merged_xml, baseline)
        reread = ET.parse(merged_xml).getroot()
        actual_signatures = []
        for node in [item for item in reread.iter() if _name(item) == 'Event']:
            graphic = _one(node, 'Graphic')
            actual_signatures.append({'event': dict(node.attrib), 'graphic': dict(graphic.attrib),
                                      'image': graphic.text, 'png_sha256': _sha256(merged / graphic.text)})
        _require(actual_signatures == event_signatures,
                 'Merged event order, coordinates, times or PNG bytes changed')
        _require(_snapshot(sup) == original_snapshot and _sha256(sup) == original_hash,
                 'Original SUP changed during bounded export')
        _require(_snapshot(baseline) == baseline_snapshot and _sha256(baseline) == baseline_hash,
                 'Full timing baseline changed during bounded export')
        if expected_xml.parent.exists():
            _require(expected_xml.parent.is_dir() and not any(expected_xml.parent.iterdir()),
                     f'Refuse to replace an unknown nonempty BDN directory: {expected_xml.parent}')
        with merged_xml.open('rb') as complete_xml:
            os.fsync(complete_xml.fileno())
        audit.update(status='verified', event_count=cursor, merged_xml_sha256=_sha256(merged_xml),
                     source_event_count=source_cursor, event_signatures=event_signatures,
                     raw_export_event_count=source_cursor, canonical_count=cursor,
                     canonical_consumed_count=cursor, source_baseline_event_count=len(full_timing),
                     exact_merged_timing_match=True)
        _json(audit_path, audit)
        # The entire verified directory appears at once. A SIGKILL cannot leave
        # orphan PNGs in the OCR input directory; a nonempty target is rejected
        # both above and by the operating system's directory-rename semantics.
        os.rename(merged, expected_xml.parent)
        audit['status'] = 'complete'
        _json(audit_path, audit)
        return expected_xml
    except BaseException as exc:
        audit.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                     failure=f'{type(exc).__name__}: {exc}', completed_entries=cursor)
        _json(audit_path, audit)
        raise
