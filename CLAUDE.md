<a id="zh-cn"></a>

# Flash-MoE：在笔记本上运行 397B 参数模型

[![中文](https://img.shields.io/badge/中文-默认-black)](#zh-cn)
[![English](https://img.shields.io/badge/English-Switch-blue)](#en)

> **[阅读论文](paper/flash_moe.pdf)**：完整技术细节、90+ 次实验，以及这套系统在 24 小时内由人类与 AI 协作完成的全过程。

这是一个纯 C / Metal 推理引擎，可以在 48GB 内存的 MacBook Pro 上运行 **Qwen3.5-397B-A17B**（397B 参数 MoE 模型），并达到 **4.4+ tokens/s** 的实际速度，同时支持较高质量输出与工具调用。

整个 209GB 模型通过自定义 Metal 计算管线从 SSD 流式读取。没有 Python 推理框架，没有重量级运行时，只有 C、Objective-C 和手写 Metal kernel。

## 结果

![Progress](progress.png)

| 配置 | tok/s | 质量 | 说明 |
|---|---:|---|---|
| 4-bit experts, FMA kernel | **4.36** | Excellent | 当前最佳方案，可用于工具调用，磁盘占用 209GB |
| 4-bit experts, baseline | 3.90 | Excellent | FMA 优化前 |
| 2-bit experts, trust OS | 5.74 | Good* | 磁盘占用 120GB，结构化输出不稳定 |
| 2-bit peak single token | 7.05 | Good* | 热缓存峰值，不适合工具调用 |

\* 2-bit 量化会破坏 JSON / tool calling 的稳定性，例如把 `"` 变成错误字符。生产默认应使用 4-bit。

## 硬件

- **机器**：MacBook Pro
- **芯片**：Apple M3 Max / M4 Pro 同类 Apple Silicon 均可参考
- **内存**：48 GB unified memory
- **SSD**：高顺序读带宽 NVMe
- **系统**：现代 macOS

## 架构概览

模型共有 60 个 Transformer 层：45 层 GatedDeltaNet 线性注意力 + 15 层标准全注意力。每层有 512 个 experts，每个 token 激活其中 K=4 个，同时还有 1 个 shared expert。隐藏维度为 4096。

### 关键技术

1. **SSD Expert Streaming**：209GB 的 expert 权重不常驻内存，而是按 token 通过 `pread()` 从 SSD 读取，仅加载当前激活的 K=4 个 experts。
2. **FMA 优化反量化 kernel**：将 `(nibble * scale + bias) * x` 改写为 `fma(nibble, scale*x, bias*x)`，更好利用 GPU FMA 单元。
3. **Metal Compute Shaders**：手写 4-bit / 2-bit matvec、SwiGLU、RMSNorm、RoPE、attention 与 MoE 融合 kernel。
4. **Deferred GPU Expert Compute**：expert 前向延迟提交，让 CPU 在 GPU 工作时准备下一层。
5. **Accelerate BLAS**：GatedDeltaNet 的线性注意力递推使用 `cblas_*` 系列。
6. **Trust the OS**：不做自定义 expert cache，直接利用 macOS page cache。

### 每层流水线（4-bit 平均约 4.28ms）

```text
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                       [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared   [0.55ms GPU]
           → CPU: softmax + topK routing              [0.003ms]
           → I/O: parallel pread K=4 experts          [2.41ms SSD]
           → CMD3: expert forward + combine + norm    [0.04ms encode, DEFERRED]
```

### Unified Memory 约束

在 Apple Silicon 上，SSD DMA 和 GPU 计算共用内存控制器，难以获得理想重叠。背景 SSD 读经常会明显拖慢 GPU，因此最终最优设计是串行的 `GPU -> SSD -> GPU` 管线。

## 快速开始

下面这套步骤对应当前仓库实际代码路径，按顺序执行可以尽量避免我们这次调试里踩过的坑。

### 0. 首次环境准备

如果是第一次在新机器上跑，建议先完成下面这些步骤：

```bash
# 1) Xcode Command Line Tools
xcode-select --install

# 2) Python 依赖 + Hugging Face CLI
python3 -m pip install -U numpy "huggingface_hub[cli]"

# 3) 拉代码并编译
git clone https://github.com/clawwangcai-dev/flash-moe.git
cd flash-moe/metal_infer
make
cd ..

# 4) 登录 Hugging Face（如果模型需要鉴权）
hf auth login

# 5) 下载模型到 Hugging Face 缓存
hf download mlx-community/Qwen3.5-397B-A17B-4bit
```

上面第 5 步默认会把模型放到 `$HOME/.cache/huggingface/hub/...`，后面的 `MODEL_DIR` 变量通常就指向这里。

### 1. 准备模型快照目录

使用 Hugging Face snapshot 目录，至少包含：

- `model.safetensors.index.json`
- `model-00001-of-....safetensors`
- `tokenizer.json`

示例：

```bash
MODEL_DIR="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/<your-snapshot-id>"
```

### 2. 生成 expert 索引

在仓库根目录执行：

```bash
cd /Volumes/SSD1T/projects/flash-moe
python3 make_expert_index.py \
  --model-dir "$MODEL_DIR" \
  --out expert_index.json
```

其中 `<your-snapshot-id>` 只是示例占位符，需要替换成你本机实际的 snapshot 目录名。

`expert_index.json` 会写入当前机器的实际 `model_path`，所以应在目标机器上现生成，不建议跨机器复制。

### 3. 打包 4-bit experts

```bash
cd /Volumes/SSD1T/projects/flash-moe
python3 repack_experts.py --index /Volumes/SSD1T/projects/flash-moe/expert_index.json
```

输出目录：

```bash
$MODEL_DIR/packed_experts/
```

成功信号：

```text
[experts] 60/60 packed layer files available
```

### 4. 可选：生成 2-bit experts

2-bit 更快，但质量更差，不适合依赖稳定 JSON / tool calling 的场景。

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
python3 repack_experts_2bit.py
```

输出目录：

```bash
$MODEL_DIR/packed_experts_2bit/
```

### 5. 导出非 expert 权重和 tokenizer

当前仓库不会自动从 Hugging Face snapshot 生成这些本地文件，需要显式执行。

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
python3 extract_weights.py --model "$MODEL_DIR" --output .
python3 export_tokenizer.py "$MODEL_DIR/tokenizer.json" tokenizer.bin
python3 export_vocab.py "$MODEL_DIR/tokenizer.json" vocab.bin
```

注意：这些脚本里保留了作者机器上的默认 snapshot 路径，所以实际使用时建议始终显式传入 `"$MODEL_DIR"`。

### 6. 检查本地推理文件

运行前请确认这些文件都在：

- `metal_infer/model_weights.bin`
- `metal_infer/model_weights.json`
- `metal_infer/tokenizer.bin`
- `metal_infer/vocab.bin`

### 7. 编译推理程序

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
make infer
```

### 8. 运行 4-bit 推理

4-bit 是推荐默认配置。

默认示例：中文

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
./infer --prompt "你好，做个自我介绍" --tokens 64
```

切换为英文：

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
./infer --prompt "Hi, introduce yourself briefly." --tokens 64
```

程序现在会优先从 `../expert_index.json` 自动读取 `model_path`。如果你想强制指定模型目录：

```bash
./infer \
  --model "$MODEL_DIR" \
  --prompt "你好，做个自我介绍" \
  --tokens 64
```

### 9. 运行 2-bit 推理

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
./infer --prompt "你好，做个自我介绍" --tokens 64 --2bit
```

英文示例：

```bash
./infer --prompt "Hi, introduce yourself briefly." --tokens 64 --2bit
```

成功信号：

- `Quant:    2-bit experts`
- `[experts] 60/60 packed layer files available`

### 10. 交互式聊天

`./chat` 是一个本地客户端，不会自己启动推理服务。先开服务端，再开聊天界面。

先在一个终端启动服务：

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
./infer --serve 8000
```

再在另一个终端启动聊天：

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
make chat
./chat
```

## 常见问题

`zsh: no such file or directory: ./infer`

- 说明你当前在仓库根目录，不在 `metal_infer/` 目录下。

`ERROR: Cannot open vocab vocab.bin`

- 在 `metal_infer/` 目录执行：

```bash
python3 export_tokenizer.py "$MODEL_DIR/tokenizer.json" tokenizer.bin
python3 export_vocab.py "$MODEL_DIR/tokenizer.json" vocab.bin
```

`ERROR: Failed to load weights`

- 先执行：

```bash
python3 extract_weights.py --model "$MODEL_DIR" --output .
```

`[experts] 0/60 packed layer files available`

- `packed_experts/` 没生成，或者 `infer` 指向了错误的模型目录。
- 重新在本机生成 `expert_index.json`，确认其中 `model_path` 正确。

`FileNotFoundError: 找不到 ... model.safetensors.index.json`

- `--model-dir` 路径写错了。
- 或者你把路径在引号内部错误换行了。

## 性能 / 质量说明

- 4-bit 是默认生产路径。
- 2-bit 主要用于速度实验。
- `--prompt` 现在走 chat template，因此中英文 prompt 都可以直接输入，无需手动拼 `<|im_start|>` 标签。

## 项目结构

```text
metal_infer/
  infer.m                # 完整推理引擎
  shaders.metal          # Metal compute kernels
  chat.m                 # 交互式聊天 TUI
  tokenizer.h            # 单头文件 BPE tokenizer
  main.m                 # MoE benchmark
  Makefile               # 构建脚本
  extract_weights.py     # 导出非-expert 权重
  repack_experts_2bit.py # 4-bit -> 2-bit expert 重打包
  train_predictor.py     # expert routing 分析
  model_weights.bin      # 非-expert 权重
  model_weights.json     # manifest
  vocab.bin              # token 显示用词表
  tokenizer.bin          # BPE tokenizer 导出结果

repack_experts.py        # 4-bit expert 打包脚本
progress.py              # 结果可视化
results.tsv              # 实验记录
```

## 做过什么尝试，哪些有效

### 保留下来的方案

| 方案 | 结果 | 影响 |
|---|---|---|
| FMA dequant kernel | GPU 计算时间 -12% | **+12% tok/s** |
| Trust OS page cache | 删掉 Metal LRU 后反而 +38% | **基础性优化** |
| GPU combine + norm in CMD3 | 避免 CPU 往返 | **流水线优化** |
| BLAS delta-net | cpu_attn 0.78ms -> 0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% | **2-bit only** |
| GPU fused attention (RoPE) | +2% | **小幅提升** |
| C BPE tokenizer | 启动 180ms vs 3500ms | **20x 启动优化** |
| Deferred CMD3 | GPU / CPU overlap | **流水线优化** |

### 放弃的方案（58 次实验中的代表）

| 方案 | 结果 | 原因 |
|---|---|---|
| LZ4 expert compression | -13% | 解压开销大于收益 |
| F_RDADVISE prefetch | 净收益 0% | SSD DMA 会拖慢 GPU |
| Temporal expert prediction | -18% | 命中率太低 |
| MLP routing predictor | 31% accuracy | 不如简单基线 |
| GPU LUT dequant kernel | -2% | 间接访问导致串行化 |
| GPU private buffer compression | -20% | blit 开销过大 |
| Spin-poll GPU wait | -23% | CPU 发热影响 GPU |
| Expert file clustering | 0% | NVMe 在这个粒度下不受益 |
| dispatch_io | -70% | dispatch_data 管理开销过高 |
| mmap expert files | -5x | 冷页 fault 太重 |
| Speculative early routing | -38% | 污染 cache |
| MTP speculative decoding | 接近持平 | MoE I/O 不像 dense 模型那样受益 |

## 安全性与资源控制

这是主力开发机，所以引擎对资源有明确约束：

- 非-expert 权重：约 5.5GB，只读 `mmap`
- Metal scratch buffers：约 200MB
- 总常驻内存：约 6GB
- 剩余大量内存留给系统和 page cache
- 不做自定义大缓存，避免 OOM 与复杂失控行为

---

<a id="en"></a>

# Flash-MoE: Running a 397B Parameter Model on a Laptop

[![中文](https://img.shields.io/badge/中文-Switch-blue)](#zh-cn)
[![English](https://img.shields.io/badge/English-Default-black)](#en)

> **[Read the paper](paper/flash_moe.pdf)** for the full technical details, 90+ experiments, and the story behind the system.

Flash-MoE is a pure C / Metal inference engine for **Qwen3.5-397B-A17B**, designed to run on Apple Silicon laptops by streaming 209GB of MoE expert weights directly from SSD.

## Results

![Progress](progress.png)

| Configuration | tok/s | Quality | Notes |
|---|---:|---|---|
| 4-bit experts, FMA kernel | **4.36** | Excellent | Best overall configuration, suitable for tool calling |
| 4-bit experts, baseline | 3.90 | Excellent | Before the FMA optimization |
| 2-bit experts, trust OS | 5.74 | Good* | Faster, but unreliable for structured output |
| 2-bit peak single token | 7.05 | Good* | Warm-cache burst, not the recommended default |

\* 2-bit quantization hurts JSON / tool-calling reliability. Use 4-bit by default.

## Quick Start

This is the shortest setup path that matches the codebase as it exists today.

### 0. First-time setup

On a new machine, this is the recommended path:

```bash
# 1) Xcode Command Line Tools
xcode-select --install

# 2) Python dependencies + Hugging Face CLI
python3 -m pip install -U numpy "huggingface_hub[cli]"

# 3) Clone and build
git clone https://github.com/clawwangcai-dev/flash-moe.git
cd flash-moe/metal_infer
make
cd ..

# 4) Login to Hugging Face if the model requires auth
hf auth login

# 5) Download the model into the Hugging Face cache
hf download mlx-community/Qwen3.5-397B-A17B-4bit
```

Step 5 usually puts the model under `$HOME/.cache/huggingface/hub/...`, which is what `MODEL_DIR` will point to below.

### 1. Prepare the model snapshot

Use a Hugging Face snapshot directory containing:

- `model.safetensors.index.json`
- `model-xxxxx-of-xxxxx.safetensors`
- `tokenizer.json`

Example:

```bash
MODEL_DIR="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/<your-snapshot-id>"
```

### 2. Build `expert_index.json`

```bash
cd /Volumes/SSD1T/projects/flash-moe
python3 make_expert_index.py \
  --model-dir "$MODEL_DIR" \
  --out expert_index.json
```

`<your-snapshot-id>` is a placeholder. Replace it with the actual snapshot directory on your machine.

Generate this file on the target machine. It stores the machine-local `model_path`.

### 3. Pack 4-bit experts

```bash
cd /Volumes/SSD1T/projects/flash-moe
python3 repack_experts.py --index /Volumes/SSD1T/projects/flash-moe/expert_index.json
```

Output directory:

```bash
$MODEL_DIR/packed_experts/
```

### 4. Optional: build 2-bit experts

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
python3 repack_experts_2bit.py
```

Output directory:

```bash
$MODEL_DIR/packed_experts_2bit/
```

### 5. Export non-expert weights and tokenizer files

The repo does not generate these files automatically from the Hugging Face snapshot. Run them explicitly:

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
python3 extract_weights.py --model "$MODEL_DIR" --output .
python3 export_tokenizer.py "$MODEL_DIR/tokenizer.json" tokenizer.bin
python3 export_vocab.py "$MODEL_DIR/tokenizer.json" vocab.bin
```

These scripts still contain author-local default paths, so in practice you should always pass `"$MODEL_DIR"` explicitly.

### 6. Verify local inference assets

Required local files for `infer`:

- `metal_infer/model_weights.bin`
- `metal_infer/model_weights.json`
- `metal_infer/tokenizer.bin`
- `metal_infer/vocab.bin`

### 7. Build `infer`

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
make infer
```

### 8. Run 4-bit inference

Default example: Chinese

```bash
./infer --prompt "你好，做个自我介绍" --tokens 64
```

Switch to English:

```bash
./infer --prompt "Hi, introduce yourself briefly." --tokens 64
```

If you want to force the model path explicitly:

```bash
./infer \
  --model "$MODEL_DIR" \
  --prompt "Hi, introduce yourself briefly." \
  --tokens 64
```

### 9. Run 2-bit inference

```bash
./infer --prompt "Hi, introduce yourself briefly." --tokens 64 --2bit
```

Expected signals:

- `Quant:    2-bit experts`
- `[experts] 60/60 packed layer files available`

### 10. Interactive chat

`./chat` is only a local client. It does not start the inference server for you.

Start the server in one terminal:

```bash
cd /Volumes/SSD1T/projects/flash-moe/metal_infer
./infer --serve 8000
```

Then start the chat client in another terminal:

```bash
make chat
./chat
```

## Common Failures

`zsh: no such file or directory: ./infer`

- You are in the repo root, not `metal_infer/`.

`ERROR: Cannot open vocab vocab.bin`

- Run:

```bash
python3 export_tokenizer.py "$MODEL_DIR/tokenizer.json" tokenizer.bin
python3 export_vocab.py "$MODEL_DIR/tokenizer.json" vocab.bin
```

`ERROR: Failed to load weights`

- Run:

```bash
python3 extract_weights.py --model "$MODEL_DIR" --output .
```

`[experts] 0/60 packed layer files available`

- `packed_experts/` was not generated, or `infer` is reading the wrong model path.

`FileNotFoundError: ... model.safetensors.index.json`

- Your `--model-dir` path is wrong, or you accidentally broke the path across lines inside the quotes.

## Notes

- 4-bit is the production default.
- 2-bit is mainly for speed experiments.
- `--prompt` now goes through the chat-template path, so both Chinese and English prompts work directly.
