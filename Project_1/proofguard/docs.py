"""Document loading: text layer -> OCR fallback -> page images; paragraph chunks + retrieval for long contracts."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
_cuad: dict | None = None
_cuad_lock = threading.Lock()


def cuad_texts() -> dict:
    global _cuad
    with _cuad_lock:
        if _cuad is None:
            raw = json.loads((DATA / "raw/cuad/CUADv1.json").read_text(encoding="utf-8"))["data"]
            _cuad = {d["title"]: "\n".join(p["context"] for p in d["paragraphs"]) for d in raw}
        return _cuad


@dataclass
class Chunk:
    id: int
    start: int
    end: int
    text: str


@dataclass
class Document:
    doc_id: str
    source: str
    text: str = ""                       # full text of the text layer (or CUAD contract)
    pages: list = field(default_factory=list)   # text per page
    path: Path | None = None
    ocr_pages: list = field(default_factory=list)
    cache_dir: Path | None = None
    max_pages: int = 4
    _chunks: list | None = None
    _images: list | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip()) if self.path and self.path.suffix.lower() == ".txt" else len(self.text.strip()) >= 200

    @property
    def is_scan(self) -> bool:
        return self.source != "cuad" and not self.has_text

    @property
    def ocr_text(self) -> str:
        return "\n\f".join(self.ocr_pages)

    @property
    def evidence_text(self) -> str:
        """Text the proof engine may check answers against: the text layer, else OCR output."""
        return self.text if self.has_text else self.ocr_text

    @property
    def evidence_kind(self) -> str:
        return "layer" if self.has_text else ("ocr" if self.ocr_text.strip() else "none")

    @property
    def chunks(self) -> list:
        if self._chunks is None:
            self._chunks = make_chunks(self.text)
        return self._chunks

    def page_of(self, offset: int) -> int:
        """1-based page number of a character offset in self.text (CUAD: always 1)."""
        if len(self.pages) <= 1:
            return 1
        acc = 0
        for i, p in enumerate(self.pages):
            acc += len(p) + 1
            if offset < acc:
                return i + 1
        return len(self.pages)

    def images(self) -> list:
        """[(page_no, data_url)] for a vision model (first max_pages pages)."""
        if self._images is None:
            self._images = render_pages(self.path, self.max_pages, self.cache_dir) if self.path else []
        return self._images


def make_chunks(text: str, target: int = 900, hard: int = 1600) -> list:
    """Paragraph chunks with offsets; tiny blocks (headings) are merged into the following block."""
    blocks = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text)]
    sent = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\d\"“])")
    out, carry = [], None
    for s, e in blocks:
        if carry is not None:
            s = carry
            carry = None
        seg = text[s:e]
        if len(seg) < 160 and e < len(text):
            carry = s
            continue
        if len(seg) <= hard:
            out.append((s, e))
            continue
        a = 0
        for m in sent.finditer(seg):
            if m.start() - a >= target:
                out.append((s + a, s + m.start()))
                a = m.end()
        out.append((s + a, e))
    if carry is not None:
        out.append((carry, len(text)))
    return [Chunk(i, s, e, text[s:e]) for i, (s, e) in enumerate(out)]


def retrieve(chunks: list, patterns: list, k: int) -> list:
    """Top-k chunks by weighted pattern hits (position order). Cheap lexical retrieval; the LLM does the reading."""
    scored = []
    for c in chunks:
        sc = sum(w * min(len(p.findall(c.text)), 2) for p, w in patterns)
        if sc > 0:
            scored.append((sc, -c.id, c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return sorted((x[2] for x in scored[:k]), key=lambda c: c.id)


def _pdf_pages(path: Path) -> list:
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True)
        if r.returncode == 0:
            pg = r.stdout.split("\f")
            if pg and not pg[-1].strip():
                pg = pg[:-1]
            return pg
    try:
        from pypdf import PdfReader
        return [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    except Exception:
        return []


def render_pages(path: Path, max_pages: int, cache_dir: Path | None) -> list:
    """Render PDF pages to JPEG data URLs (pdftoppm); image files are passed through."""
    if path is None or not path.exists():
        return []
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        mime = {"png": "image/png", "webp": "image/webp"}.get(path.suffix.lower().lstrip("."), "image/jpeg")
        return [(1, f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}")]
    if not shutil.which("pdftoppm"):
        return []
    tag = hashlib.sha256(path.read_bytes() + str(max_pages).encode()).hexdigest()[:20]
    out_dir = (cache_dir or Path("/tmp")) / "pages" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not list(out_dir.glob("p-*.jpg")):
        subprocess.run(["pdftoppm", "-jpeg", "-jpegopt", "quality=80", "-r", "110", "-f", "1", "-l", str(max_pages), str(path), str(out_dir / "p")],
                       capture_output=True)
    res = []
    for f in sorted(out_dir.glob("p-*.jpg")):
        n = int(re.search(r"p-(\d+)", f.name).group(1))
        res.append((n, f"data:image/jpeg;base64,{base64.b64encode(f.read_bytes()).decode()}"))
    return res[:max_pages]


def _ocr(path: Path, max_pages: int, cache_dir: Path) -> list:
    """Independent second reader for scans (tesseract, if installed)."""
    if not shutil.which("tesseract") or not shutil.which("pdftoppm"):
        return []
    tag = hashlib.sha256(path.read_bytes() + str(max_pages).encode()).hexdigest()[:20]
    out_dir = cache_dir / "ocr" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = out_dir / "ocr.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    pages = []
    if not list(out_dir.glob("o-*.png")):
        subprocess.run(["pdftoppm", "-png", "-r", "200", "-f", "1", "-l", str(max_pages), str(path), str(out_dir / "o")], capture_output=True)
    for f in sorted(out_dir.glob("o-*.png")):
        r = subprocess.run(["tesseract", str(f), "-", "--psm", "6"], capture_output=True, text=True)
        pages.append(r.stdout if r.returncode == 0 else "")
    cached.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def load_document(rec: dict, cache_dir: Path, max_pages: int = 4, want_ocr: bool = True) -> Document:
    src = rec.get("source_dataset") or rec["doc_id"].split(":", 1)[0]
    if src == "cuad":
        title = rec["doc_id"].split(":", 1)[1]
        txt = cuad_texts()[title]
        return Document(rec["doc_id"], src, text=txt, pages=[txt], cache_dir=cache_dir, max_pages=max_pages)
    path = (DATA / rec["pages"][0]).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError("Document path must remain inside data/")
    doc = Document(rec["doc_id"], src, path=path, cache_dir=cache_dir, max_pages=max_pages)
    if path.suffix.lower() == ".txt":
        doc.text = path.read_text(encoding="utf-8-sig")
        doc.pages = [doc.text]
    if path.suffix.lower() == ".pdf":
        pages = _pdf_pages(path)
        doc.pages, doc.text = pages, "\n\f".join(pages)
        if not doc.has_text and want_ocr:
            try:
                doc.ocr_pages = _ocr(path, max_pages, cache_dir)
            except Exception:
                doc.ocr_pages = []
    return doc
