"""Extract DeepSeek PoW WASM module from chat.deepseek.com.

Usage:
    python extract_wasm.py              # auto-fetch latest WASM
    python extract_wasm.py --check      # check if current WASM still valid

DeepSeek loads its PoW WASM at runtime. This script:
1. Fetches the main page HTML
2. Finds the WASM URL in JS source or known paths
3. Downloads and saves the WASM module
4. Verifies it exports the expected functions
"""
import re
import sys
import hashlib
from pathlib import Path

import httpx
import wasmtime

WASM_DIR = Path(__file__).parent / "wasm"
WASM_PATH = WASM_DIR / "sha3_wasm_bg.wasm"

DEEPSEEK_BASE = "https://chat.deepseek.com"
# Known WASM paths — try these in order
KNOWN_PATHS = [
    "/static/wasm/sha3_wasm_bg.wasm",
    "/static/wasm/deepseek_hash_bg.wasm",
]
# JS pattern to find WASM URL
WASM_URL_RE = re.compile(r'["\']([^"\']+\.wasm)["\']')


def fetch_page() -> str:
    """Fetch DeepSeek main page to find WASM URLs in JS."""
    with httpx.Client(http2=True, timeout=30) as client:
        r = client.get(
            DEEPSEEK_BASE,
            headers={
                "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "accept": "text/html,*/*",
            },
        )
        r.raise_for_status()
        return r.text


def find_wasm_urls(html: str) -> list[str]:
    """Extract .wasm URLs from HTML/JS."""
    urls = set()
    for m in WASM_URL_RE.finditer(html):
        url = m.group(1)
        if url.startswith("/"):
            url = f"{DEEPSEEK_BASE}{url}"
        elif not url.startswith("http"):
            url = f"{DEEPSEEK_BASE}/{url}"
        urls.add(url)
    return list(urls)


def verify_wasm(path: Path) -> dict | None:
    """Verify WASM exports the expected PoW functions. Returns export info."""
    if not path.exists():
        return None
    try:
        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, path.read_bytes())
        store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        instance = linker.instantiate(store, module)
        exports = instance.exports(store)

        funcs = []
        for exp in exports:
            try:
                exports[exp]
                funcs.append(exp)
            except Exception:
                pass

        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "exports": funcs,
            "has_solve": "wasm_solve" in funcs,
        }
    except Exception as e:
        return {"error": str(e)}


def download_wasm(url: str) -> bytes | None:
    """Download a WASM file."""
    try:
        with httpx.Client(http2=True, timeout=30) as client:
            r = client.get(
                url,
                headers={
                    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    "origin": "https://chat.deepseek.com",
                    "referer": "https://chat.deepseek.com/",
                },
            )
            if r.status_code == 200 and r.content[:4] == b"\x00asm":
                return r.content
    except Exception:
        pass
    return None


def main():
    check_only = "--check" in sys.argv

    print("=== DeepSeek WASM Extractor ===\n")

    # 1. Try known paths first (fast path)
    print("Trying known WASM paths...")
    for path in KNOWN_PATHS:
        url = f"{DEEPSEEK_BASE}{path}"
        data = download_wasm(url)
        if data:
            print(f"  Found: {url} ({len(data)} bytes)")
            WASM_DIR.mkdir(parents=True, exist_ok=True)
            WASM_PATH.write_bytes(data)
            info = verify_wasm(WASM_PATH)
            if info and info.get("has_solve"):
                print(f"  Valid: sha256={info['sha256']}, exports={info['exports']}")
                return
            else:
                print(f"  Invalid WASM: {info}")
        else:
            print(f"  Not found: {url}")

    # 2. Fall back to parsing the page
    print("\nFetching DeepSeek page to find WASM URL...")
    try:
        html = fetch_page()
        urls = find_wasm_urls(html)
        print(f"  Found {len(urls)} WASM URLs in page source")
        for url in urls:
            print(f"  Trying: {url}")
            data = download_wasm(url)
            if data:
                print(f"  Downloaded: {url} ({len(data)} bytes)")
                WASM_DIR.mkdir(parents=True, exist_ok=True)
                WASM_PATH.write_bytes(data)
                info = verify_wasm(WASM_PATH)
                if info and info.get("has_solve"):
                    print(f"  Valid: sha256={info['sha256']}")
                    return
                else:
                    print(f"  Invalid: {info}")
    except Exception as e:
        print(f"  Page fetch failed: {e}")

    # 3. Report current state
    print("\n=== Current WASM Status ===")
    info = verify_wasm(WASM_PATH)
    if info:
        print(f"  File: {info['path']}")
        print(f"  SHA256: {info['sha256']}")
        print(f"  Exports: {info['exports']}")
        print(f"  PoW solver: {'OK' if info['has_solve'] else 'MISSING'}")
    else:
        print(f"  Error: {info}")
        print("\n  Manual extraction needed:")
        print("  1. Open https://chat.deepseek.com in Chrome")
        print("  2. DevTools → Network → filter '.wasm'")
        print("  3. Right-click WASM file → 'Save as...' → save to wasm/")
        print("  4. Run: python extract_wasm.py --check")


if __name__ == "__main__":
    main()
