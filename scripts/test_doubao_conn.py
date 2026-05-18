"""
Doubao gateway connectivity test.

Run from repo root:
    python -m scripts.test_doubao_conn          # uses .env
    python -m scripts.test_doubao_conn --tier lite
    python -m scripts.test_doubao_conn --image https://example.com/cat.jpg

Exit code 0 = success, 1 = failure.
"""
import argparse
import os
import sys
import time

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure repo root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.llm_client import LLMClient, LLMConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["pro", "lite"], default="pro")
    parser.add_argument("--prompt", default="请回复 OK")
    parser.add_argument("--image", default=None, help="optional image URL to test multimodal input")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = LLMConfig.from_env()
    if not cfg.app_id or not cfg.app_key:
        print("[FAIL] DOUBAO_APP_ID / DOUBAO_APP_KEY not loaded. Did you `cp .env.example .env`?")
        sys.exit(1)

    print(f"[info] host        = {cfg.host}")
    print(f"[info] tier        = {args.tier}")
    print(f"[info] model       = {cfg.model_pro if args.tier == 'pro' else cfg.model_lite}")
    print(f"[info] app_id      = {cfg.app_id[:6]}...{cfg.app_id[-4:]}")

    cli = LLMClient(cfg=cfg, mock=False)

    if args.image:
        messages = [{"role": "user",
                     "content": [
                         {"type": "text", "value": args.prompt + " (describe the image briefly)"},
                         {"type": "image_url", "value": args.image},
                     ]}]
    else:
        messages = [{"role": "user", "content": args.prompt}]

    t0 = time.time()
    try:
        out = cli.chat(messages, tier=args.tier, use_cache=not args.no_cache)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        sys.exit(1)
    dt = time.time() - t0

    print(f"[ok]   latency     = {dt:.2f}s")
    print(f"[ok]   answer      = {out!r}")
    print("[PASS] Doubao gateway is reachable.")


if __name__ == "__main__":
    main()
