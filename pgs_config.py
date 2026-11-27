"""Runtime paths for the queue and its OCR subprocesses."""

import os
from pathlib import Path
import shutil

CODE_DIR = Path(__file__).resolve().parent
TMP_ROOT = Path(os.environ.get('TMPDIR', str(Path.home() / '.cache' / 'extract-pgs'))).expanduser().resolve()
WORK = Path(os.environ.get('PGS_WORK_DIR', str(TMP_ROOT / 'queue'))).expanduser().resolve()
MANIFEST = Path(os.environ.get('PGS_MANIFEST', str(WORK / 'queue.md'))).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get('PGS_MODEL_ROOT', str(Path.home() / '.local' / 'share' / 'extract-pgs' / 'models'))).expanduser().resolve()
SECONV = shutil.which(os.environ.get('PGS_SECONV', 'seconv'))
TESSDATA_ENG = Path(os.environ.get('PGS_TESSDATA_ENG', '/usr/share/tesseract-ocr/5/tessdata/eng.traineddata')).expanduser().resolve()
