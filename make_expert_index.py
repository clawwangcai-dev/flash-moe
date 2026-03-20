#!/usr/bin/env python3
import argparse
import json
import re
import struct
from pathlib import Path

COMPONENTS = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
]

LAYOUTS = [
    {
        "name": "mlx_experts",
        "layer_re": re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(.+)$"),
        "tensor_key": "model.layers.{layer}.mlp.experts.{component}",
    },
    {
        "name": "mlx_switch_mlp",
        "layer_re": re.compile(
            r"^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.(.+)$"
        ),
        "tensor_key": (
            "language_model.model.layers.{layer}.mlp.switch_mlp.{component}"
        ),
    },
]

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "U32": 4,
    "I32": 4,
    "U16": 2,
    "I16": 2,
    "U8": 1,
    "I8": 1,
}


def load_safetensors_header(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def detect_layout(weight_map):
    for layout in LAYOUTS:
        layers = set()
        for key in weight_map:
            m = layout["layer_re"].match(key)
            if m and m.group(2) in COMPONENTS:
                layers.add(int(m.group(1)))
        if layers:
            return layout, sorted(layers)
    return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF snapshot directory")
    ap.add_argument("--out", default="expert_index.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    index_path = model_dir / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(f"找不到 {index_path}")

    with index_path.open() as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]

    layout, layers = detect_layout(weight_map)

    if not layers:
        raise RuntimeError(
            "没在 weight_map 里找到支持的 experts/switch_mlp 键，"
            "检查是不是下载了正确的 MLX 模型。"
        )

    header_cache = {}
    expert_reads = {}

    for layer in layers:
        layer_key = str(layer)
        expert_reads[layer_key] = {}

        for comp in COMPONENTS:
            tensor_key = layout["tensor_key"].format(layer=layer, component=comp)
            if tensor_key not in weight_map:
                raise KeyError(f"缺少 tensor: {tensor_key}")

            fname = weight_map[tensor_key]
            fpath = model_dir / fname

            if fname not in header_cache:
                header_cache[fname] = load_safetensors_header(fpath)

            header_len, header = header_cache[fname]

            if tensor_key not in header:
                raise KeyError(f"{fname} 的 safetensors header 里没有 {tensor_key}")

            meta = header[tensor_key]
            shape = meta["shape"]
            dtype = meta["dtype"]
            start, end = meta["data_offsets"]

            if dtype not in DTYPE_BYTES:
                raise ValueError(f"未知 dtype {dtype} for {tensor_key}")

            tensor_nbytes = end - start
            abs_offset = 8 + header_len + start

            # 这里假设第 1 维就是 512 个 experts
            if not shape or shape[0] != 512:
                raise ValueError(
                    f"{tensor_key} 的 shape={shape}，不是预期的 [512, ...]，"
                    "需要手动检查这个模型的布局。"
                )

            expert_stride = tensor_nbytes // 512
            if tensor_nbytes % 512 != 0:
                raise ValueError(f"{tensor_key} 的总字节数不能被 512 整除")

            expert_reads[layer_key][comp] = {
                "file": fname,
                "abs_offset": abs_offset,
                "expert_stride": expert_stride,
                "expert_size": expert_stride,
            }

    out = {
        "model_path": str(model_dir),
        "expert_reads": expert_reads,
    }

    out_path = Path(args.out).expanduser().resolve()
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"写好了: {out_path}")
    print(f"model_path: {model_dir}")
    print(f"layout: {layout['name']}")
    print(f"layers: {layers[0]}..{layers[-1]} ({len(layers)} layers)")


if __name__ == "__main__":
    main()
