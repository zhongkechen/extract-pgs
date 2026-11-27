"""Exact whole-baseline alignment for over-split BDN cues in a batch.

The time-only planner authorizes no image merge. align_batch additionally
requires identical complete RGBA pixels, placement and non-time XML metadata.
Aggregations require an already-exported fragment SUP and its caller-verified
SHA256: seconv 5.3.0 writes BDN Forced=false, so BDN alone cannot prove original
PGS forced flags. This module reads PCS headers only; it invokes no converter
or OCR. The Pulp instantaneous-event filter must use its separate path.
"""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import xml.etree.ElementTree as ET

from PIL import Image
from pgs_config import TMP_ROOT
from validation import parse_srt, read_bdn


SCHEMA = 'bdn-exact-timing-alignment-v1'
MAX_IMAGE_RGBA_BYTES = 128 * 1024 * 1024


class AlignmentError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AlignmentError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def rows(value):
    parsed = parse_srt(value) if isinstance(value, (str, os.PathLike)) else list(value)
    return [dict(item) for item in parsed]


def times(value):
    return [(item['start_ms'], item['end_ms']) for item in value]


def _validate_rows(value, label, local=False):
    require(value, f'{label} is empty')
    first = 1 if local else value[0].get('index')
    require(type(first) is int and first >= 1, f'{label} has invalid first index')
    for position, item in enumerate(value):
        require(type(item.get('index')) is int and item['index'] == first + position,
                f'{label} indexes are not consecutive')
        start, end = item.get('start_ms'), item.get('end_ms')
        require(type(start) is int and type(end) is int and 0 <= start < end,
                f'{label} must contain integer, positive-duration timestamps; zero events require their own filter')
        require(position == 0 or start >= value[position - 1]['start_ms'],
                f'{label} start timestamps run backwards')


def plan_batch_timing(chunk_timing, remaining_full_baseline, *, zero_event_filter_active=False):
    """Return an exact contiguous partition, never permission to merge images.

    chunk indexes must start at 1. remaining_full_baseline is the still-unused
    suffix of the whole baseline, retaining its original consecutive indexes.
    Each whole cue may be split into two or more chunk cues. Every group is
    independently certified by align_batch. Reverse/coarser chunk cues and
    incomplete cues are rejected.
    """
    require(not zero_event_filter_active, 'Do not apply aggregation to the instantaneous-event filter branch')
    chunk, full = rows(chunk_timing), rows(remaining_full_baseline)
    _validate_rows(chunk, 'Chunk', local=True)
    _validate_rows(full, 'Remaining whole baseline')
    cursor, consumed, aggregates, groups, timing = 0, 0, 0, [], []
    while cursor < len(chunk):
        require(consumed < len(full), 'Chunk extends beyond remaining whole baseline')
        target = full[consumed]
        begin = cursor
        require(chunk[cursor]['start_ms'] == target['start_ms'],
                'Chunk start differs from the next exact whole-baseline start')
        while True:
            require(cursor < len(chunk), 'Chunk ends inside a whole-baseline cue')
            current = chunk[cursor]
            if cursor > begin:
                require(current['start_ms'] == chunk[cursor - 1]['end_ms'],
                        'Sub-event partition has a gap or overlap')
            require(current['end_ms'] <= target['end_ms'],
                    'Chunk is coarser than whole baseline; reverse splitting is unsupported')
            cursor += 1
            if current['end_ms'] == target['end_ms']:
                break
        count = cursor - begin
        if count > 1:
            aggregates += 1
        consumed += 1
        timing.append({**target, 'index': consumed, 'full_baseline_index': target['index']})
        groups.append({'output_index': consumed, 'full_baseline_index': target['index'],
                       'chunk_indices': list(range(begin + 1, cursor + 1)),
                       'start_ms': target['start_ms'], 'end_ms': target['end_ms'],
                       'action': 'aggregate_exact_partition' if count > 1 else 'preserve_one_to_one'})
    require(times(timing) == times(full[:consumed]), 'Planned output differs from the whole-baseline prefix')
    return {'schema': SCHEMA, 'status': 'time-only-plan-not-image-certified',
            'chunk_count': len(chunk), 'consumed_count': consumed,
            'full_first_index': full[0]['index'], 'full_last_index': full[consumed - 1]['index'],
            'aggregation_count': aggregates, 'groups': groups, 'timing': timing,
            'chunk_timing': chunk, 'whole_prefix': full[:consumed],
            'all_chunk_events_consumed': True, 'exact_whole_prefix_times': True,
            'requires_runtime_pixel_and_source_property_proof': bool(aggregates)}


def _name(element):
    return element.tag.rsplit('}', 1)[-1]


def _one(root, name, predicate=lambda element: True):
    result = [element for element in root.iter() if _name(element) == name and predicate(element)]
    require(len(result) == 1, f'Expected exactly one {name}')
    return result[0]


def _properties(node):
    """Every non-time property, including unknown children/Graphic attributes."""
    def visit(element):
        attrs = dict(element.attrib)
        if element is node:
            attrs.pop('InTC', None)
            attrs.pop('OutTC', None)
        content = '' if _name(element) == 'Graphic' else (element.text or '').strip()
        return {'tag': element.tag, 'attributes': attrs, 'text': content,
                'children': [visit(child) for child in element]}
    return visit(node)


def _rgba(path):
    with Image.open(path) as picture:
        require(picture.format == 'PNG', 'Graphic must reference a real PNG')
        require(0 < picture.width * picture.height * 4 <= MAX_IMAGE_RGBA_BYTES,
                'PNG exceeds the bounded decoded-image budget')
        return picture.size, picture.convert('RGBA').tobytes()


def _source_compositions(source_sup, expected_sha256, chunk):
    """Bind visible PCS flags/coordinates to each unmerged chunk cue exactly.

    This is a metadata proof, not a second PGS decoder. If a fragment has
    already merged visible PCS events, its flags cannot be mapped here and the
    aggregation fails closed. No timestamp rounding is introduced.
    """
    source_sup = Path(source_sup).resolve()
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
            'Aggregation requires the caller-verified fragment SUP SHA256')
    before, actual_sha = snapshot(source_sup), sha(source_sup)
    require(actual_sha == expected_sha256, 'Fragment SUP hash differs from the caller-verified bytes')
    display_sets, current = [], None
    with source_sup.open('rb') as source:
        while header := source.read(13):
            offset = source.tell() - len(header)
            require(len(header) == 13, 'Truncated SUP segment header')
            magic, pts, dts, kind, size = struct.unpack('>2sIIBH', header)
            payload = source.read(size)
            require(magic == b'PG' and len(payload) == size, 'Truncated or invalid SUP segment')
            require(kind in (0x14, 0x15, 0x16, 0x17, 0x80), 'Unsupported PGS segment')
            if kind == 0x16:
                require(current is None and size >= 11, 'Invalid PCS/END sequence')
                if display_sets:
                    require(pts >= display_sets[-1]['pts_90khz'], 'PCS timestamps run backwards')
                    display_sets[-1]['end_90khz'] = pts
                width, height = struct.unpack_from('>HH', payload)
                objects, cursor = [], 11
                for _ in range(payload[10]):
                    require(cursor + 8 <= size, 'Truncated PCS object')
                    oid, window, flags, x, y = struct.unpack_from('>HBBHH', payload, cursor)
                    cursor += 8
                    require(flags & 0x3f == 0, 'Unsupported PCS object flags')
                    crop = None
                    if flags & 0x80:
                        require(cursor + 8 <= size, 'Truncated PCS crop')
                        crop = list(struct.unpack_from('>HHHH', payload, cursor))
                        cursor += 8
                    objects.append({'object_id': oid, 'window_id': window, 'flags': flags,
                                    'forced': bool(flags & 0x40), 'x': x, 'y': y, 'crop': crop})
                require(cursor == size, 'Unexpected PCS trailing bytes')
                current = {'pts_90khz': pts, 'end_90khz': None, 'pcs_offset': offset,
                           'pcs_payload_sha256': hashlib.sha256(payload).hexdigest(),
                           'properties': {'canvas_width': width, 'canvas_height': height, 'objects': objects},
                           'forced': any(item['forced'] for item in objects)}
            else:
                require(current is not None, 'PGS segment outside a complete display set')
                if kind == 0x80:
                    require(size == 0, 'Nonempty PGS END')
                    display_sets.append(current)
                    current = None
    require(current is None and display_sets and not display_sets[-1]['properties']['objects'],
            'Fragment must end with a complete cleared display set')
    visible = [item for item in display_sets if item['properties']['objects']]
    require(len(visible) == len(chunk), 'Original PCS flags cannot be mapped one-to-one to chunk cues')
    for index, (item, cue) in enumerate(zip(visible, chunk), 1):
        require(item['pts_90khz'] == cue['start_ms'] * 90 and item['end_90khz'] == cue['end_ms'] * 90,
                f'Original PCS flag/time mapping is ambiguous at chunk cue {index}')
        item['chunk_index'] = index
    require(snapshot(source_sup) == before and sha(source_sup) == actual_sha,
            'Fragment SUP changed during source-property validation')
    return {'path': str(source_sup), 'sha256': actual_sha, 'visible_compositions': visible,
            'exact_chunk_timing_match': True, 'decode_performed': False}


def _link(source, destination, checksum):
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f'Refuse to replace output image: {destination}')
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != 18:
            raise
        with Path(source).open('rb') as origin, destination.open('xb') as target:
            shutil.copyfileobj(origin, target)
    require(sha(destination) == checksum, 'Copied PNG bytes changed')


def align_batch(raw_xml, chunk_timing, remaining_full_baseline, *, output_dir, evidence_dir,
                source_sup=None, expected_source_sup_sha256=None, zero_event_filter_active=False):
    """Return xml/timing/consumed_count/evidence after strict runtime proof.

    evidence_dir must be a new directory outside raw batch/images and filtered
    output. Keep it during cleanup; it retains original XML/PNG and full maps.
    Caller advances its WHOLE-baseline cursor by consumed_count, not by the
    larger raw chunk count. Pulp zero filtering must bypass this function.
    """
    plan = plan_batch_timing(chunk_timing, remaining_full_baseline,
                            zero_event_filter_active=zero_event_filter_active)
    raw_xml, output_dir, evidence_dir = map(lambda p: Path(p).resolve(), (raw_xml, output_dir, evidence_dir))

    require(TMP_ROOT != Path('/tmp') and not TMP_ROOT.is_relative_to('/tmp'), 'TMPDIR must not be /tmp')
    for path in (output_dir, evidence_dir):
        require(path.is_relative_to(TMP_ROOT), 'Alignment outputs must remain under TMPDIR')
        require(not path.exists(), 'Use fresh output/evidence directories for each attempt')
    require(not output_dir.is_relative_to(raw_xml.parent) and not raw_xml.parent.is_relative_to(output_dir),
            'Filtered output must be separate from original images')
    require(not evidence_dir.is_relative_to(raw_xml.parent.parent) and
            not evidence_dir.is_relative_to(output_dir) and not output_dir.is_relative_to(evidence_dir),
            'Alignment evidence must stay outside raw batch/images and filtered output')
    original_bytes, original_snapshot = raw_xml.read_bytes(), snapshot(raw_xml)
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    root = ET.fromstring(original_bytes)
    nodes = [node for node in root.iter() if _name(node) == 'Event']
    container = _one(root, 'Events', lambda node: node.get('NumberofEvents') is None)
    require(list(container) == nodes, 'Unexpected BDN event nesting')
    # This is the existing validator, with its existing one-frame BDN-format
    # tolerance. The independent millisecond mapping above remains exact.
    parsed = read_bdn(raw_xml, plan['chunk_timing'])
    require(len(nodes) == len(parsed) == plan['chunk_count'], 'Raw XML/timing counts differ')
    records = []
    for index, (node, item) in enumerate(zip(nodes, parsed), 1):
        graphic = _one(node, 'Graphic')
        image_path = Path(item['image_path'])
        size, rgba = _rgba(image_path)
        require(size == (item['width'], item['height']), 'PNG/Graphic dimensions differ')
        forced_text = node.get('Forced')
        forced = None
        if forced_text is not None:
            require(forced_text.lower() in ('true', 'false'), 'Invalid BDN Forced attribute')
            forced = forced_text.lower() == 'true'
        records.append({'chunk_index': index, 'image_path': str(image_path),
                        'png_sha256': sha(image_path), 'rgba_sha256': hashlib.sha256(rgba).hexdigest(),
                        'size': list(size), 'x': item['x'], 'y': item['y'], 'forced': forced,
                        'event_attributes': dict(node.attrib), 'graphic_attributes': dict(graphic.attrib),
                        'non_time_properties': _properties(node)})
    source_proof = None
    if plan['aggregation_count']:
        require(source_sup is not None, 'Aggregation requires original PGS forced/position evidence')
        source_proof = _source_compositions(source_sup, expected_source_sup_sha256, plan['chunk_timing'])
    mapping = []
    for group in plan['groups']:
        indices = [index - 1 for index in group['chunk_indices']]
        first = records[indices[0]]
        merged = len(indices) > 1
        if merged:
            require(first['forced'] is not None, 'Aggregation requires an explicit BDN Forced marker')
            for left, right in zip(indices, indices[1:]):
                require(parsed[left]['bdn_end_ms'] == parsed[right]['bdn_start_ms'],
                        'Original BDN sub-event frame boundaries have a gap or overlap')
            reference_size, reference_rgba = _rgba(first['image_path'])
            source_reference = source_proof['visible_compositions'][indices[0]]
            for index in indices:
                item = records[index]
                require(item['non_time_properties'] == first['non_time_properties'] and
                        (item['size'], item['x'], item['y'], item['forced']) ==
                        (first['size'], first['x'], first['y'], first['forced']),
                        'Sub-events differ in dimensions, coordinates, Forced or other non-time attributes')
                actual_size, actual_rgba = _rgba(item['image_path'])
                require(actual_size == reference_size and actual_rgba == reference_rgba,
                        'Sub-events are not identical complete RGBA images')
                source_item = source_proof['visible_compositions'][index]
                require(source_item['properties'] == source_reference['properties'],
                        'Original PGS object geometry/flags differ across sub-events')
                require(source_item['forced'] == item['forced'],
                        'BDN Forced does not match original PGS flags; cannot certify aggregation')
            del reference_rgba, actual_rgba
        mapping.append({**group, 'selected_chunk_index': group['chunk_indices'][0],
                        'selected_png_sha256': first['png_sha256'],
                        'all_sub_events_rgba_identical': True if merged else None,
                        'source_properties_identical': True if merged else None,
                        'source_forced_matches_bdn': True if merged else None})
    # No output is materialized until all potentially lossy merges pass.
    output_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    archive_raw = evidence_dir / 'raw'
    archive_raw.mkdir()
    archived_xml = archive_raw / raw_xml.name
    archived_xml.write_bytes(original_bytes)
    for node, record in zip(nodes, records):
        graphic = _one(node, 'Graphic')
        relative = Path((graphic.text or '').strip())
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe raw Graphic archive path')
        destination = archive_raw / relative
        require(destination != archived_xml, 'Graphic path collides with archived XML')
        if destination.exists():
            require(sha(destination) == record['png_sha256'], 'Duplicate Graphic path has different image bytes')
        else:
            _link(record['image_path'], destination, record['png_sha256'])
        record['archived_png'] = str(destination)
    if not plan['aggregation_count']:
        aligned_xml = raw_xml  # exact original XML and image references
    else:
        aligned_root = deepcopy(root)
        aligned_container = _one(aligned_root, 'Events', lambda node: node.get('NumberofEvents') is None)
        for child in list(aligned_container):
            aligned_container.remove(child)
        for group in mapping:
            first_index, last_index = group['chunk_indices'][0] - 1, group['chunk_indices'][-1] - 1
            copied = deepcopy(nodes[first_index])
            if len(group['chunk_indices']) > 1:
                copied.set('OutTC', nodes[last_index].get('OutTC'))
            graphic = _one(copied, 'Graphic')
            image_name = f'aligned-{group["output_index"]:06d}.png'
            _link(records[first_index]['image_path'], output_dir / image_name, records[first_index]['png_sha256'])
            graphic.text = image_name
            aligned_container.append(copied)
        summary = _one(aligned_root, 'Events', lambda node: node.get('NumberofEvents') is not None)
        summary.set('NumberofEvents', str(plan['consumed_count']))
        summary.set('FirstEventInTC', aligned_container[0].get('InTC'))
        summary.set('LastEventOutTC', aligned_container[-1].get('OutTC'))
        aligned_xml = output_dir / 'index.xml'
        ET.ElementTree(aligned_root).write(aligned_xml, encoding='utf-8', xml_declaration=True)
    output_entries = read_bdn(aligned_xml, plan['timing'])
    result_nodes = [node for node in ET.parse(aligned_xml).getroot().iter() if _name(node) == 'Event']
    require(len(result_nodes) == len(output_entries) == plan['consumed_count'], 'Aligned counts differ from whole prefix')
    for node, item, group in zip(result_nodes, output_entries, mapping):
        origin = nodes[group['selected_chunk_index'] - 1]
        expected_attributes = dict(origin.attrib)
        if len(group['chunk_indices']) > 1:
            expected_attributes['OutTC'] = nodes[group['chunk_indices'][-1] - 1].get('OutTC')
        require(dict(node.attrib) == expected_attributes and _properties(node) == _properties(origin),
                'Aligned event metadata changed unexpectedly')
        require(sha(item['image_path']) == group['selected_png_sha256'], 'Aligned PNG bytes changed')
    require(times(output_entries) == times(plan['whole_prefix']), 'Aligned times are not the exact whole-baseline prefix')
    require(snapshot(raw_xml) == original_snapshot and sha(raw_xml) == original_hash and sha(archived_xml) == original_hash,
            'Original/archived XML changed')
    for record in records:
        require(sha(record['image_path']) == record['png_sha256'] == sha(record['archived_png']),
                'Original or archived PNG changed')
    if source_proof:
        require(sha(source_proof['path']) == source_proof['sha256'], 'Original fragment SUP changed')
    evidence = {'schema': SCHEMA, 'status': 'complete',
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'plan': plan, 'mapping': mapping, 'images': records, 'source_property_proof': source_proof,
                'raw_xml': str(raw_xml), 'raw_xml_sha256': original_hash,
                'archived_raw_xml': str(archived_xml), 'aligned_xml': str(aligned_xml),
                'aligned_xml_sha256': sha(aligned_xml), 'raw_count': plan['chunk_count'],
                'output_count': len(output_entries), 'consumed_count': plan['consumed_count'],
                'all_raw_events_mapped_once': True, 'exact_whole_prefix_times': True,
                'ordinary_one_to_one_attributes_and_png_bytes_preserved': True,
                'strict_read_bdn_passed': True, 'time_tolerance_changed': False,
                'zero_event_semantics_changed': False, 'preserve_evidence_during_cleanup': True,
                'module_sha256': sha(__file__)}
    report_path = evidence_dir / 'alignment-evidence.json'
    with report_path.open('x', encoding='utf-8') as output:
        json.dump(evidence, output, ensure_ascii=False, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    return {'xml': aligned_xml, 'timing': plan['timing'], 'consumed_count': plan['consumed_count'],
            'entries': output_entries, 'evidence': report_path}
