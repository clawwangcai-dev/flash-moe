#!/usr/bin/env python3
"""Export HuggingFace tokenizer.json to the vocab.bin format used by infer.m.

Format:
  uint32 num_entries
  uint32 max_id
  repeated num_entries times:
    uint16 byte_len
    byte[byte_len] token_utf8
"""

import json
import struct
import sys
from pathlib import Path


DEFAULT_TOKENIZER = (
    "/Volumes/SSD1T/projects/hf_cache/hub/"
    "models--mlx-community--Qwen3.5-397B-A17B-4bit/"
    "snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/tokenizer.json"
)


def main():
    tok_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOKENIZER)
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "vocab.bin")

    with tok_path.open("r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    entries = {int(v): k for k, v in tokenizer["model"]["vocab"].items()}
    for tok in tokenizer.get("added_tokens", []):
        entries[int(tok["id"])] = tok["content"]

    max_id = max(entries)
    missing = [i for i in range(max_id + 1) if i not in entries]
    if missing:
        raise ValueError(
            f"token ids are not contiguous: missing {len(missing)} ids, first few={missing[:8]}"
        )

    with out_path.open("wb") as f:
        f.write(struct.pack("<I", len(entries)))
        f.write(struct.pack("<I", max_id))
        for token_id in range(max_id + 1):
            b = entries[token_id].encode("utf-8")
            if len(b) > 0xFFFF:
                raise ValueError(f"token {token_id} too long: {len(b)} bytes")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    print(f"Exported {out_path}")
    print(f"  entries: {len(entries)}")
    print(f"  max_id:  {max_id}")
    print(f"  size:    {out_path.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
