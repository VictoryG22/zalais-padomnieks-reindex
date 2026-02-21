import os
import re
import sys
import json
import time
import shutil
import hashlib
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# CONFIG
# =========================
RESOURCES_URL = "https://biofruitnet.eu/resources/"
# Anchor is not sent to server, but we keep it for clarity:
PRACTICE_ANCHOR_URL = "https://biofruitnet.eu/resources/#practice-abstracts"

DATA_DIR = pathlib.Path("data")
PDF_DIR = DATA_DIR / "pdfs"

DB_DIR = pathlib.Path("db")  # artifact will upload this folder
COLLECTION_NAME = "biofruitnet_practice_abstracts"

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Retrieval quality
MIN_PAGE_TEXT_LEN = 50

# Networking
TIMEOUT = 90
SLEEP_BETWEEN_DOWNLOADS = 0.0  # set 0.2 if you want to be polite

USER_AGENT = "Mozilla/5.0 (compatible; BiofruitnetReindexBot/1.0; +https://github.com/)"

# Embeddings model
EMBED_MODEL = "text-embedding-3-small"


# =========================
# Helpers
# =========================
def ensure_openai_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it as environment variable."
        )


def fetch_html(url: str, session: requests.Session) -> str:
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def extract_pdf_urls(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    pdfs = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = urljoin(base_url, href)
        if re.search(r"\.pdf(\?|$)", full, flags=re.IGNORECASE):
            pdfs.append(full)

    # unique, keep order
    seen = set()
    uniq = []
    for u in pdfs:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def filename_from_url(url: str) -> str:
    return unquote(urlparse(url).path.split("/")[-1])


def safe_filename_from_url(url: str) -> str:
    name = filename_from_url(url)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    if name.strip() in ["", ".pdf"]:
        h = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        name = f"doc_{h}.pdf"
    return name


def download_pdf(url: str, out_dir: pathlib.Path, session: requests.Session) -> Tuple[pathlib.Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = safe_filename_from_url(url)
    fp = out_dir / fn

    if fp.exists() and fp.stat().st_size > 0:
        return fp, "SKIPPED_EXISTS"

    r = session.get(url, timeout=TIMEOUT, stream=True)
    r.raise_for_status()

    tmp = fp.with_suffix(fp.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    os.replace(tmp, fp)

    # quick signature check
    with open(fp, "rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        # keep file but warn
        print(f"⚠️ Warning: downloaded file does not look like a PDF header: {fp.name} head={head}")

    return fp, "DOWNLOADED"


def pdf_to_documents(pdf_path: pathlib.Path, pdf_url: Optional[str], splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Convert PDF pages into chunked LangChain Documents with metadata:
    - source: filename
    - page: 1-based page number
    - chunk: chunk index within page
    - url: original pdf url (if available)
    """
    reader = PdfReader(str(pdf_path))
    base = pdf_path.name
    docs: List[Document] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # normalize whitespace
        text = re.sub(r"\s+\n", "\n", text).strip()
        text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])

        if len(text) < MIN_PAGE_TEXT_LEN:
            continue

        chunks = splitter.split_text(text)
        for j, ch in enumerate(chunks):
            meta = {
                "source": base,
                "page": i + 1,
                "chunk": j,
            }
            if pdf_url:
                meta["url"] = pdf_url
            docs.append(Document(page_content=ch, metadata=meta))

    return docs


def build_chroma(docs: List[Document], db_dir: pathlib.Path, collection: str):
    if db_dir.exists():
        shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(db_dir),
        collection_name=collection
    )
    # persist may be optional depending on version
    try:
        vs.persist()
    except Exception:
        pass

    return vs


# =========================
# Main
# =========================
def main():
    ensure_openai_key()

    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print(f"Fetching resources page: {RESOURCES_URL}")
    html = fetch_html(RESOURCES_URL, session)

    pdf_urls = extract_pdf_urls(html, RESOURCES_URL)
    print(f"Found PDF URLs: {len(pdf_urls)}")
    if not pdf_urls:
        raise RuntimeError("No PDF links found on the resources page.")

    # Download PDFs
    url_map: Dict[str, str] = {}  # filename -> url
    results = []
    for u in tqdm(pdf_urls, desc="Downloading PDFs"):
        try:
            fp, status = download_pdf(u, PDF_DIR, session)
            url_map[fp.name] = u
            results.append(status)
            if SLEEP_BETWEEN_DOWNLOADS:
                time.sleep(SLEEP_BETWEEN_DOWNLOADS)
        except Exception as e:
            print("ERROR downloading:", u, repr(e))

    print("Download summary:",
          "downloaded=", results.count("DOWNLOADED"),
          "skipped=", results.count("SKIPPED_EXISTS"))

    # Build docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Local PDFs: {len(pdf_files)}")

    all_docs: List[Document] = []
    for p in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            all_docs.extend(pdf_to_documents(p, url_map.get(p.name), splitter))
        except Exception as e:
            print("PDF parse error:", p.name, repr(e))

    print(f"Total chunks (documents): {len(all_docs)}")
    if not all_docs:
        raise RuntimeError("No extracted text chunks. PDFs may be scanned images or extraction failed.")

    # Save manifest (useful for debugging/demo)
    manifest = {
        "resources_url": RESOURCES_URL,
        "practice_anchor_url": PRACTICE_ANCHOR_URL,
        "pdf_count": len(pdf_files),
        "chunk_count": len(all_docs),
        "collection": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DATA_DIR / "url_map.json").write_text(json.dumps(url_map, indent=2), encoding="utf-8")

    # Build Chroma DB
    print(f"Building Chroma DB at: {DB_DIR} (collection: {COLLECTION_NAME})")
    build_chroma(all_docs, DB_DIR, COLLECTION_NAME)

    print("✅ Done.")
    print("Manifest:", (DATA_DIR / "manifest.json").resolve())
    print("DB dir:", DB_DIR.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Reindex failed:", repr(e))
        sys.exit(1)
