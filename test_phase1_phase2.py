"""
test_phase1_phase2.py
---------------------
Lumen Reader — Phase 1 & Phase 2 Test Suite
Run from project root:
    python test_phase1_phase2.py

Requires at least one PDF in data/uploads/
All tests are self-contained and print clear PASS / FAIL results.
"""

import os
import sys
import json
import time
import traceback

# ── Make src importable ───────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"

results = []

def run_test(name, fn):
    print(f"\n  Testing: {BOLD}{name}{RESET}")
    t0 = time.time()
    try:
        fn()
        elapsed = round(time.time() - t0, 3)
        print(f"  {PASS}  ({elapsed}s)")
        results.append((name, True, None))
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        print(f"  {FAIL}  ({elapsed}s)")
        print(f"         {RED}{e}{RESET}")
        results.append((name, False, str(e)))

def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


# ════════════════════════════════════════════════════════════
#  PHASE 1 TESTS — Environment
# ════════════════════════════════════════════════════════════

section("PHASE 1 · Environment & Dependencies")

# ── T1: Core imports ──────────────────────────────────────

def test_import_fitz():
    import fitz
    assert hasattr(fitz, "open"), "fitz.open missing"

def test_import_langchain():
    import langchain
    assert langchain.__version__, "langchain version missing"

def test_import_langgraph():
    import langgraph
    assert langgraph.__version__, "langgraph version missing"

def test_import_qdrant():
    import qdrant_client
    assert qdrant_client.__version__, "qdrant_client version missing"

def test_import_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    assert SentenceTransformer, "SentenceTransformer import failed"

def test_import_sklearn():
    import sklearn
    assert sklearn.__version__, "sklearn version missing"

def test_import_streamlit():
    import streamlit
    assert streamlit.__version__, "streamlit version missing"

def test_import_dotenv():
    from dotenv import load_dotenv
    assert load_dotenv is not None

def test_import_pyttsx3():
    import pyttsx3
    assert pyttsx3, "pyttsx3 import failed"

def test_import_gtts():
    from gtts import gTTS
    assert gTTS, "gTTS import failed"

def test_import_torch():
    import torch
    assert torch.__version__, "torch version missing"

run_test("fitz (PyMuPDF)",           test_import_fitz)
run_test("langchain",                 test_import_langchain)
run_test("langgraph",                 test_import_langgraph)
run_test("qdrant-client",             test_import_qdrant)
run_test("sentence-transformers",     test_import_sentence_transformers)
run_test("scikit-learn",              test_import_sklearn)
run_test("streamlit",                 test_import_streamlit)
run_test("python-dotenv",             test_import_dotenv)
run_test("pyttsx3",                   test_import_pyttsx3)
run_test("gTTS",                      test_import_gtts)
run_test("torch",                     test_import_torch)

# ── T2: .env file ────────────────────────────────────────

section("PHASE 1 · .env & Config")

def test_env_exists():
    assert os.path.exists(".env"), ".env not found at project root"

def test_env_keys():
    from dotenv import dotenv_values
    env = dotenv_values(".env")
    required = [
        "GROQ_API_KEY",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "EMBEDDING_MODEL",
        "RERANKER_MODEL",
        "GROQ_MODEL",
    ]
    missing = [k for k in required if k not in env or not env[k]]
    assert not missing, f"Missing or empty .env keys: {missing}"

def test_groq_key_not_placeholder():
    from dotenv import dotenv_values
    env = dotenv_values(".env")
    key = env.get("GROQ_API_KEY", "")
    assert key != "your_groq_key_here", (
        "GROQ_API_KEY is still the placeholder. "
        "Replace it with your real key from console.groq.com"
    )

run_test(".env file exists",          test_env_exists)
run_test(".env has all required keys", test_env_keys)
run_test("GROQ_API_KEY is real key",  test_groq_key_not_placeholder)

# ── T3: Folder structure ─────────────────────────────────

section("PHASE 1 · Folder Structure")

def test_folders():
    required_dirs = [
        "src/ingestion",
        "src/labelling",
        "src/vectordb",
        "src/retrieval",
        "src/generation",
        "src/accessibility",
        "src/ui",
        "data/uploads",
        "models",
    ]
    missing = [d for d in required_dirs if not os.path.isdir(d)]
    assert not missing, f"Missing directories: {missing}"

def test_stub_files():
    required_files = [
        "src/__init__.py",
        "src/ingestion/__init__.py",
        "src/ingestion/loader.py",
        "src/ingestion/chunker.py",
        "src/labelling/__init__.py",
        "src/labelling/labeller.py",
        "src/vectordb/__init__.py",
        "src/vectordb/store.py",
        "src/retrieval/__init__.py",
        "src/retrieval/graph.py",
        "src/generation/__init__.py",
        "src/generation/answer.py",
        "src/accessibility/__init__.py",
        "src/accessibility/tts.py",
        "src/accessibility/stt.py",
        "src/ui/app.py",
        "run.py",
        "requirements.txt",
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    assert not missing, f"Missing files: {missing}"

def test_ingestion_files_nonempty():
    for f in ["src/ingestion/loader.py", "src/ingestion/chunker.py"]:
        size = os.path.getsize(f)
        assert size > 100, f"{f} is too small ({size} bytes) — may not have been written"

run_test("All directories exist",       test_folders)
run_test("All stub files exist",        test_stub_files)
run_test("Ingestion files non-empty",   test_ingestion_files_nonempty)

# ── T4: Qdrant connectivity ─────────────────────────────

section("PHASE 1 · Qdrant Connection")

def test_qdrant_connection():
    from dotenv import dotenv_values
    from qdrant_client import QdrantClient
    env = dotenv_values(".env")
    url = env.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=5)
    info = client.get_collections()
    assert info is not None, "Qdrant returned None"

run_test("Qdrant reachable at localhost:6333", test_qdrant_connection)


# ════════════════════════════════════════════════════════════
#  PHASE 2 TESTS — Ingestion Pipeline
# ════════════════════════════════════════════════════════════

section("PHASE 2 · Module Imports")

def test_loader_import():
    from ingestion.loader import load_document
    assert callable(load_document)

def test_chunker_import():
    from ingestion.chunker import chunk_document
    assert callable(chunk_document)

def test_ingestion_package_import():
    from ingestion import load_document, chunk_document
    assert callable(load_document)
    assert callable(chunk_document)

run_test("loader.load_document importable",   test_loader_import)
run_test("chunker.chunk_document importable", test_chunker_import)
run_test("ingestion package __init__ exports", test_ingestion_package_import)

# ── T5: Loader unit tests ────────────────────────────────

section("PHASE 2 · Loader Unit Tests")

def test_loader_rejects_missing_file():
    from ingestion.loader import load_document
    try:
        load_document("data/uploads/does_not_exist_xyz.pdf")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # correct

def test_loader_rejects_bad_extension():
    from ingestion.loader import load_document
    # Create a temp file with unsupported extension
    tmp = "data/uploads/_test_bad.xyz"
    with open(tmp, "w") as f:
        f.write("dummy")
    try:
        load_document(tmp)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # correct
    finally:
        os.remove(tmp)

def test_loader_clean_text():
    """Test the text cleaner handles edge cases correctly."""
    from ingestion.loader import _clean_text
    dirty = "Hello   world\r\n\r\n\r\nSecond   paragraph\x00"
    clean = _clean_text(dirty)
    assert "\x00" not in clean, "Null bytes not removed"
    assert "\r" not in clean,   "Carriage returns not removed"
    assert "  " not in clean,   "Double spaces not collapsed"
    assert "\n\n\n" not in clean, "Triple newlines not collapsed"

run_test("Loader raises FileNotFoundError",  test_loader_rejects_missing_file)
run_test("Loader raises ValueError bad ext", test_loader_rejects_bad_extension)
run_test("Text cleaner handles edge cases",  test_loader_clean_text)

# ── T6: Loader with real PDF ─────────────────────────────

section("PHASE 2 · Loader with Real File")

def _find_test_pdf():
    uploads = "data/uploads"
    for f in os.listdir(uploads):
        if f.lower().endswith(".pdf"):
            return os.path.join(uploads, f)
    return None

TEST_PDF = _find_test_pdf()

if TEST_PDF:
    print(f"  {INFO} Using test file: {os.path.basename(TEST_PDF)}")
else:
    print(f"  {WARN} No PDF found in data/uploads/ — skipping real-file tests")
    print(f"  {WARN} Drop any PDF into data/uploads/ and re-run to enable these tests")

def test_loader_pdf_returns_text():
    if not TEST_PDF:
        raise AssertionError("No PDF in data/uploads/ — add a file and re-run")
    from ingestion.loader import load_document
    text, meta = load_document(TEST_PDF)
    assert isinstance(text, str),  "text is not a string"
    assert len(text) > 50,         f"text too short ({len(text)} chars) — may be empty PDF"
    assert meta["filename"],       "filename missing from metadata"
    assert meta["page_count"] > 0, "page_count is 0"

def test_loader_pdf_metadata_keys():
    if not TEST_PDF:
        raise AssertionError("No PDF in data/uploads/ — add a file and re-run")
    from ingestion.loader import load_document
    _, meta = load_document(TEST_PDF)
    required_keys = ["filename", "extension", "file_size_kb", "page_count"]
    missing = [k for k in required_keys if k not in meta]
    assert not missing, f"Missing metadata keys: {missing}"

def test_loader_pdf_no_null_bytes():
    if not TEST_PDF:
        raise AssertionError("No PDF in data/uploads/ — add a file and re-run")
    from ingestion.loader import load_document
    text, _ = load_document(TEST_PDF)
    assert "\x00" not in text, "Null bytes found in cleaned text"

def test_loader_pdf_page_markers():
    if not TEST_PDF:
        raise AssertionError("No PDF in data/uploads/ — add a file and re-run")
    from ingestion.loader import load_document
    import re
    text, meta = load_document(TEST_PDF)
    if meta["page_count"] > 1:
        markers = re.findall(r"\[Page \d+\]", text)
        assert len(markers) > 0, "Multi-page PDF has no [Page N] markers"

run_test("Loader returns text + metadata",  test_loader_pdf_returns_text)
run_test("Metadata has all required keys",  test_loader_pdf_metadata_keys)
run_test("No null bytes in output text",    test_loader_pdf_no_null_bytes)
run_test("[Page N] markers in multi-page",  test_loader_pdf_page_markers)

# ── T7: Chunker unit tests ───────────────────────────────

section("PHASE 2 · Chunker Unit Tests")

def test_chunker_basic():
    from ingestion.chunker import chunk_document
    sample = "This is a test paragraph.\n\nThis is another test paragraph with more content to split on boundaries.\n\nAnd a third one here."
    chunks = chunk_document(sample, filename="test.txt", chunk_size=80, chunk_overlap=10)
    assert isinstance(chunks, list), "chunk_document did not return a list"
    assert len(chunks) > 0,         "No chunks produced"

def test_chunker_chunk_keys():
    from ingestion.chunker import chunk_document
    sample = "A" * 600  # force at least one chunk
    chunks = chunk_document(sample, filename="test.txt")
    required_keys = ["chunk_id", "text", "filename", "chunk_index", "page_hint", "start_char", "end_char", "word_count"]
    for key in required_keys:
        assert key in chunks[0], f"Missing key in chunk: '{key}'"

def test_chunker_ids_are_unique():
    from ingestion.chunker import chunk_document
    long_text = "\n\n".join([f"Paragraph number {i} with some content to fill the space." for i in range(30)])
    chunks = chunk_document(long_text, filename="test.txt")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate chunk IDs found: {len(ids)} total, {len(set(ids))} unique"

def test_chunker_ids_deterministic():
    from ingestion.chunker import chunk_document
    text = "\n\n".join([f"Deterministic paragraph {i}." for i in range(20)])
    chunks_a = chunk_document(text, filename="same.txt")
    chunks_b = chunk_document(text, filename="same.txt")
    ids_a = [c["chunk_id"] for c in chunks_a]
    ids_b = [c["chunk_id"] for c in chunks_b]
    assert ids_a == ids_b, "Chunk IDs are not deterministic — same input produced different IDs"

def test_chunker_respects_min_length():
    from ingestion.chunker import chunk_document
    # Mix of very short and normal paragraphs
    text = "Hi.\n\nThis is a normal paragraph with enough content to pass the minimum chunk length filter.\n\nOk.\n\nAnother proper paragraph that should definitely be included in the output chunks."
    chunks = chunk_document(text, filename="test.txt", min_chunk_len=50)
    for chunk in chunks:
        assert len(chunk["text"]) >= 50, f"Chunk shorter than min_chunk_len: '{chunk['text'][:40]}'"

def test_chunker_word_count_accurate():
    from ingestion.chunker import chunk_document
    text = "\n\n".join([f"Word count test paragraph number {i} here." for i in range(10)])
    chunks = chunk_document(text, filename="test.txt")
    for chunk in chunks:
        actual = len(chunk["text"].split())
        assert chunk["word_count"] == actual, (
            f"word_count mismatch: stored={chunk['word_count']}, actual={actual}"
        )

run_test("Chunker returns list of chunks",    test_chunker_basic)
run_test("Each chunk has all required keys",  test_chunker_chunk_keys)
run_test("All chunk IDs are unique",          test_chunker_ids_are_unique)
run_test("Chunk IDs are deterministic",       test_chunker_ids_deterministic)
run_test("Min chunk length filter works",     test_chunker_respects_min_length)
run_test("word_count is accurate",            test_chunker_word_count_accurate)

# ── T8: Full pipeline test with real PDF ─────────────────

section("PHASE 2 · Full Pipeline (Loader → Chunker)")

def test_full_pipeline():
    if not TEST_PDF:
        raise AssertionError("No PDF in data/uploads/ — add a file and re-run")
    from ingestion import load_document, chunk_document

    raw_text, meta = load_document(TEST_PDF)
    chunks = chunk_document(raw_text, filename=meta["filename"])

    assert len(chunks) > 0, "Pipeline produced 0 chunks"

    # Spot check first chunk
    c = chunks[0]
    assert c["filename"] == meta["filename"], "Filename mismatch in chunk"
    assert c["chunk_index"] == 0,             "First chunk index should be 0"
    assert len(c["text"]) > 0,               "First chunk text is empty"
    assert c["chunk_id"],                     "chunk_id is empty"

    # Print a nice summary
    avg_words = sum(c["word_count"] for c in chunks) // len(chunks)
    print(f"\n         {INFO} File      : {meta['filename']}")
    print(f"         {INFO} Pages     : {meta['page_count']}")
    print(f"         {INFO} Chars     : {len(raw_text):,}")
    print(f"         {INFO} Chunks    : {len(chunks)}")
    print(f"         {INFO} Avg words : {avg_words}")
    print(f"         {INFO} Sample ID : {chunks[0]['chunk_id']}")

def test_chunk_page_hints_in_range():
    if not TEST_PDF:
        raise AssertionError("No PDF in data/uploads/ — add a file and re-run")
    from ingestion import load_document, chunk_document
    raw_text, meta = load_document(TEST_PDF)
    chunks = chunk_document(raw_text, filename=meta["filename"])
    page_count = meta["page_count"]
    for c in chunks:
        assert c["page_hint"] <= page_count, (
            f"page_hint {c['page_hint']} exceeds document page count {page_count}"
        )

run_test("Full pipeline produces valid chunks",   test_full_pipeline)
run_test("Page hints within document page range", test_chunk_page_hints_in_range)


# ════════════════════════════════════════════════════════════
#  FINAL REPORT
# ════════════════════════════════════════════════════════════

section("TEST SUMMARY")

total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n  Total  : {total}")
print(f"  {GREEN}Passed : {passed}{RESET}")
if failed:
    print(f"  {RED}Failed : {failed}{RESET}")
    print(f"\n  {BOLD}Failed tests:{RESET}")
    for name, ok, err in results:
        if not ok:
            print(f"    {FAIL} {name}")
            print(f"           → {err}")
else:
    print(f"  {GREEN}{BOLD}All tests passed!{RESET}")

print()
if failed == 0:
    print(f"  {GREEN}{BOLD}✅ Phase 1 & Phase 2 verified. Ready for Phase 3.{RESET}")
elif passed >= total * 0.8:
    print(f"  {YELLOW}{BOLD}⚠️  Most tests passed. Fix the failures above before Phase 3.{RESET}")
else:
    print(f"  {RED}{BOLD}❌ Too many failures. Review Phase 1 or Phase 2 setup.{RESET}")

print()
