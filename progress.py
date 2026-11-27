"""Sequential PGS batch progress helpers; importing this module makes no writes."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


from pgs_config import MANIFEST, TMP_ROOT, WORK

LOCK = WORK / "queue.lock"
ENTRY = re.compile(r'^- \[(?P<done>[ xX])\] \*\*(?P<title>.*?)\*\* \| 状态：(?P<status>.*?) \| 完成日期：(?P<date>.*?) \| 备注：(?P<note>.*)$', re.M)
RECORD_MARKER = '<!-- pgs-auto-record -->'
CURRENT_MARKER = '<!-- pgs-auto-current -->'


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def _one_line(value):
    return ' '.join(str(value).replace('\r', '\n').splitlines()).strip()


def _code_value(line, prefix):
    match = re.match(re.escape(prefix) + r'(?P<fence>`+)(?P<value>.*?)(?P=fence)(?!`)', line)
    if not match:
        raise ValueError(f'无法解析进度清单中的路径：{line[:180]}')
    value = match['value']
    # The inventory renderer pads code spans containing a backtick or edge space.
    if value.startswith(' ') and value.endswith(' ') and value.strip():
        value = value[1:-1]
    return value


def _valid_key(key):
    value = str(key).rstrip('/')
    parts = PurePosixPath(value).parts
    if len(parts) != 2 or any(part in {'', '.', '..'} for part in value.split('/')):
        raise ValueError(f'不是清单中的电影目录键：{key}')
    return value


def parse_text(text):
    """Parse an inventory string. Offsets are internal and valid for this text only."""
    entries = list(ENTRY.finditer(text))
    movies = []
    seen = set()
    for index, match in enumerate(entries):
        end = entries[index + 1].start() if index + 1 < len(entries) else len(text)
        block = text[match.end():end]
        directory_lines = [line for line in block.splitlines() if line.startswith('  目录：')]
        if len(directory_lines) != 1:
            raise ValueError(f'电影条目的目录行数量不正确：{match["title"]}')
        key = _valid_key(_code_value(directory_lines[0], '  目录：'))
        if key in seen:
            raise ValueError(f'电影目录键重复：{key}')
        seen.add(key)
        sources = [_code_value(line, '  来源：') for line in block.splitlines()
                   if line.startswith('  来源：')]
        for source in sources:
            parts = PurePosixPath(source).parts
            if not source.startswith(key + '/') or '..' in parts:
                raise ValueError(f'PGS 来源未归属对应电影目录：{source}')
        movies.append({
            'key': key,
            'title': re.sub(r'\\([\\`*_{}\[\]()#+.!|>~-])', r'\1', match['title']),
            'completed': match['done'].lower() == 'x',
            'status': match['status'],
            'date': match['date'],
            'note': match['note'],
            'sources': sources,
            '_start': match.start(), '_heading_end': match.end(), '_end': end,
        })
    if not movies:
        raise ValueError('进度清单没有可解析的电影条目')
    return movies


def parse_movies(path=MANIFEST):
    """Return movies in the Markdown's current order (read only)."""
    with Path(path).open('r', encoding='utf-8') as source:
        fcntl.flock(source.fileno(), fcntl.LOCK_SH)
        return parse_text(source.read())


def render_update(text, key, status, note, completed=False, *, allow_reopen=False, timestamp=None):
    """Pure transformation used by update_movie; also convenient for dry runs."""
    key = _valid_key(key)
    status, note = _one_line(status), _one_line(note)
    if not status:
        raise ValueError('状态不可为空')
    if not completed and status == '已完成':
        raise ValueError('标记已完成时必须显式传 completed=True')
    timestamp = timestamp or _now()
    movies = parse_text(text)
    target = next((item for item in movies if item['key'] == key), None)
    if target is None:
        raise KeyError(key)
    if target['completed'] and not completed and not allow_reopen:
        raise ValueError('拒绝自动撤销已完成标记；若确需重开请显式传 allow_reopen=True')
    old_heading = text[target['_start']:target['_heading_end']]
    match = ENTRY.fullmatch(old_heading)
    date = timestamp[:10] if completed else ('—' if allow_reopen else target['date'])
    heading_note = target['note']
    if heading_note.strip() in {'', '—'}:
        heading_note = '详见自动转换记录'
    heading = (f'- [{"x" if completed else " "}] **{match["title"]}** | 状态：{status}'
               f' | 完成日期：{date} | 备注：{heading_note}')
    body = text[target['_heading_end']:target['_end']]
    managed = f'  自动转换记录：{timestamp}；{note or status} {RECORD_MARKER}'
    lines = body.splitlines(keepends=True)
    managed_indices = [i for i, line in enumerate(lines) if RECORD_MARKER in line]
    if len(managed_indices) > 1:
        raise ValueError(f'存在重复自动转换记录行：{key}')
    if managed_indices:
        i = managed_indices[0]
        lines[i] = managed + ('\n' if lines[i].endswith('\n') else '')
        body = ''.join(lines)
    else:
        # Append the managed paragraph directly after the heading, before user content.
        body = '\n\n' + managed + '\n' + body.lstrip('\n')
    result = text[:target['_start']] + heading + body + text[target['_end']:]
    updated_movies = parse_text(result)
    done = sum(item['completed'] for item in updated_movies)
    total = len(updated_movies)
    summary = re.compile(r'^- (?:初始|当前)进度：\*\*\d+\s*/\s*\d+\*\*', re.M)
    result, count = summary.subn(f'- 当前进度：**{done} / {total}**', result, count=1)
    if count != 1:
        raise ValueError('未找到唯一的顶部进度统计')
    current = '无' if completed else key
    current_line = (f'- 顺序转换：当前处理：{current}；最近更新：{key}（{status}）；'
                    f'更新时间：{timestamp}。 {CURRENT_MARKER}')
    lines = result.splitlines(keepends=True)
    marked = [i for i, line in enumerate(lines) if CURRENT_MARKER in line]
    if len(marked) > 1:
        raise ValueError('顶部自动处理状态行重复')
    if marked:
        lines[marked[0]] = current_line + '\n'
    else:
        position = next(i for i, line in enumerate(lines) if line.startswith('- 当前进度：'))
        lines.insert(position + 1, current_line + '\n')
    result = ''.join(lines)
    if len(parse_text(result)) != len(movies):
        raise ValueError('更新意外改变了电影条目数')
    return result


def update_movie(key, status, note, completed=False, *, path=MANIFEST, allow_reopen=False):
    """Update one movie and the top summary, preserving all other user content.

    The file is locked and written in place because TMPDIR is on another filesystem.
    A single recovery snapshot per Markdown path is retained inside this batch's TMPDIR.
    """
    path = Path(path)
    snapshot = WORK / ('progress-before-' + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16] + '.md')
    with path.open('r+', encoding='utf-8', newline='') as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        before = os.fstat(output.fileno())
        original = output.read()
        updated = render_update(original, key, status, note, completed, allow_reopen=allow_reopen)
        try:
            with snapshot.open('x', encoding='utf-8', newline='') as recovery:
                recovery.write(original)
                recovery.flush()
                os.fsync(recovery.fileno())
        except FileExistsError:
            pass
        actual = path.stat()
        if (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            raise RuntimeError('Markdown 在加锁读取后被其它编辑器改变，本次未写入')
        output.seek(0)
        output.write(updated)
        output.truncate()
        output.flush()
        os.fsync(output.fileno())
        output.seek(0)
        if output.read() != updated:
            raise IOError(f'Markdown 写入校验失败；恢复快照：{snapshot}')
    return next(item for item in parse_text(updated) if item['key'] == _valid_key(key))


def acquire_batch_lock(path=LOCK):
    """Acquire a nonblocking process lock. KEEP the returned handle for the whole queue."""
    path = Path(path)
    if not path.resolve().is_relative_to(TMP_ROOT):
        raise ValueError('批次锁必须位于 TMPDIR')
    handle = path.open('a+', encoding='utf-8')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('已有 PGS 转换批次在运行，拒绝同时转换另一部电影') from None
    handle.seek(0)
    handle.truncate()
    handle.write(f'pid={os.getpid()}\nstarted={_now()}\n')
    handle.flush()
    os.fsync(handle.fileno())
    return handle


@contextmanager
def batch_lock(path=LOCK):
    handle = acquire_batch_lock(path)
    try:
        yield handle
    finally:
        handle.close()


def atomic_write_json(path, data):
    """Atomically replace task state, only under TMPDIR; never writes a media sidecar."""
    path = Path(path)
    if not path.resolve().is_relative_to(TMP_ROOT):
        raise ValueError('任务状态文件必须位于 TMPDIR')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', prefix=path.name + '.',
                                         suffix='.new', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
