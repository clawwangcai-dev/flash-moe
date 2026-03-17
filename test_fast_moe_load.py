"""Test fast_moe_load C extension — FRESH mx.array approach.

Verifies:
1. init() with full parameters (workers, layers, K, components, packed_dir, expert_size)
2. load_and_assemble() returns NEW dicts with FRESH mx.arrays every call
3. Data correctness: bit-exact match against direct pread
4. BF16 view pairing works (scales/biases returned as bfloat16)
5. Timing: target <50ms for 240 experts (60 layers x 4 experts x 9 components)

Uses the real 397B packed expert files.
"""

import time
import json
import os
import numpy as np
import mlx.core as mx


# --- Layout ---
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/"
    "snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"
)
PACKED_DIR = os.path.join(MODEL_PATH, "packed_experts")
LAYOUT_PATH = os.path.join(PACKED_DIR, "layout.json")

with open(LAYOUT_PATH) as f:
    layout = json.load(f)

expert_size = layout["expert_size"]
num_layers = layout["num_layers"]
num_experts = layout["num_experts"]
components = layout["components"]
K = 4  # active experts per token

DTYPE_MAP = {
    'U32': 'uint32',
    'BF16': 'uint16',  # stored as uint16, .view(bfloat16) later
    'F32': 'float32',
    'F16': 'float16',
}

# Build component specs for C extension
fml_components = []
for comp in components:
    fml_components.append({
        'name': comp['name'],
        'offset': comp['offset'],
        'size': comp['size'],
        'shape': comp['shape'],
        'dtype': DTYPE_MAP[comp['dtype']],
        'needs_bf16_view': comp['dtype'] == 'BF16',
    })


def test_init_and_stats():
    """Test that init() with full parameters works and stats are correct."""
    import fast_moe_load

    fast_moe_load.init(
        num_workers=8,
        num_layers=3,  # just 3 layers for test
        K=K,
        components=fml_components,
        packed_dir=PACKED_DIR,
        expert_size=expert_size,
    )

    s = fast_moe_load.stats()
    assert s['initialized'] == 1, f"Expected initialized=1, got {s['initialized']}"
    assert s['num_workers'] == 8, f"Expected 8 workers, got {s['num_workers']}"
    assert s['num_layers'] == 3, f"Expected 3 layers, got {s['num_layers']}"
    assert s['K'] == K, f"Expected K={K}, got {s['K']}"
    assert s['staging_size'] > 0, f"Expected staging_size > 0, got {s['staging_size']}"

    print(f"[PASS] init_and_stats: initialized with 3 layers, staging={s['staging_size']/(1024*1024):.1f} MB")

    fast_moe_load.shutdown()
    return True


def test_fresh_arrays():
    """Test that load_and_assemble returns FRESH arrays every call."""
    import fast_moe_load

    test_layers = 2

    fast_moe_load.init(
        num_workers=8,
        num_layers=test_layers,
        K=K,
        components=fml_components,
        packed_dir=PACKED_DIR,
        expert_size=expert_size,
    )

    routing = [
        (0, [10, 42, 100, 7]),
        (1, [255, 0, 128, 64]),
    ]

    # First call
    result1 = fast_moe_load.load_and_assemble(routing)
    assert isinstance(result1, list), f"Expected list, got {type(result1)}"
    assert len(result1) == test_layers, f"Expected {test_layers} layers, got {len(result1)}"

    # Grab a reference to first call's array
    arr1 = result1[0][components[0]['name']]
    arr1_copy = mx.array(arr1)  # copy the data
    mx.eval(arr1_copy)

    # Second call with DIFFERENT routing
    routing2 = [
        (0, [200, 150, 50, 1]),
        (1, [33, 77, 200, 150]),
    ]
    result2 = fast_moe_load.load_and_assemble(routing2)

    # The arrays should be DIFFERENT objects (fresh allocation)
    arr2 = result2[0][components[0]['name']]

    # And they should have different data (different experts loaded)
    mx.eval(arr2)
    mx.eval(arr1_copy)
    same = mx.array_equal(arr1_copy, arr2)
    mx.eval(same)
    assert not same.item(), "FRESH arrays should have different data for different routing"

    # The OLD result1 should still have its original data (not overwritten!)
    # This is the key difference from the old pre-allocated approach.
    mx.eval(arr1)
    old_still_valid = mx.array_equal(arr1, arr1_copy)
    mx.eval(old_still_valid)
    assert old_still_valid.item(), "Old result arrays should still be valid (not overwritten)"

    print(f"[PASS] fresh_arrays: confirmed FRESH mx.arrays each call, old data preserved")

    fast_moe_load.shutdown()
    return True


def test_data_correctness():
    """Test that load_and_assemble fills arrays with correct data (bit-exact)."""
    import fast_moe_load

    test_layers = 3

    fast_moe_load.init(
        num_workers=8,
        num_layers=test_layers,
        K=K,
        components=fml_components,
        packed_dir=PACKED_DIR,
        expert_size=expert_size,
    )

    routing = [
        (0, [10, 42, 100, 7]),
        (1, [255, 0, 128, 64]),
        (2, [33, 77, 200, 150]),
    ]

    result = fast_moe_load.load_and_assemble(routing)
    assert isinstance(result, list)
    assert len(result) == test_layers

    # Verify data by reading the same experts via direct pread
    for entry_idx, (layer_idx, expert_indices) in enumerate(routing):
        layer_dict = result[entry_idx]
        assert isinstance(layer_dict, dict), f"Entry {entry_idx}: expected dict"
        assert len(layer_dict) == len(components), (
            f"Entry {entry_idx}: expected {len(components)} components, got {len(layer_dict)}")

        packed_file = os.path.join(PACKED_DIR, f"layer_{layer_idx:02d}.bin")
        fd = os.open(packed_file, os.O_RDONLY)

        for slot, eidx in enumerate(expert_indices):
            expert_offset = eidx * expert_size

            for comp in components:
                comp_name = comp['name']
                comp_offset = comp['offset']
                comp_size = comp['size']

                # Read reference data via pread
                ref_bytes = os.pread(fd, comp_size, expert_offset + comp_offset)

                # Get the stacked array and extract slot
                stacked_arr = layer_dict[comp_name]

                # Check shape
                expected_shape = tuple([K] + comp['shape'])
                assert stacked_arr.shape == expected_shape, (
                    f"Entry {entry_idx}, {comp_name}: shape {stacked_arr.shape} != {expected_shape}")

                # Check dtype
                if comp['dtype'] == 'BF16':
                    assert stacked_arr.dtype == mx.bfloat16, (
                        f"Entry {entry_idx}, {comp_name}: dtype {stacked_arr.dtype} != bfloat16")
                    slot_arr = stacked_arr[slot].view(mx.uint16)
                elif comp['dtype'] == 'U32':
                    assert stacked_arr.dtype == mx.uint32, (
                        f"Entry {entry_idx}, {comp_name}: dtype {stacked_arr.dtype} != uint32")
                    slot_arr = stacked_arr[slot]
                else:
                    slot_arr = stacked_arr[slot]

                # Convert reference bytes to array for comparison
                if comp['dtype'] == 'U32':
                    ref_np = np.frombuffer(ref_bytes, dtype=np.uint32).reshape(comp['shape'])
                    ref_mx = mx.array(ref_np)
                elif comp['dtype'] == 'BF16':
                    ref_np = np.frombuffer(ref_bytes, dtype=np.uint16).reshape(comp['shape'])
                    ref_mx = mx.array(ref_np)
                else:
                    ref_np = np.frombuffer(ref_bytes, dtype=np.uint8)
                    slot_arr = slot_arr.flatten()
                    ref_mx = mx.array(ref_np)

                match = mx.array_equal(slot_arr, ref_mx)
                mx.eval(match)
                assert match.item(), (
                    f"Data mismatch: entry={entry_idx}, layer={layer_idx}, "
                    f"slot={slot}, expert={eidx}, comp={comp_name}")

        os.close(fd)

    print(f"[PASS] data_correctness: {len(routing)} layers, K={K}, all 9 components bit-exact")

    s = fast_moe_load.stats()
    print(f"       Stats: io={s['total_io_ms']:.1f}ms, create={s['total_create_ms']:.1f}ms")

    fast_moe_load.shutdown()
    return True


def test_timing():
    """Benchmark: target <50ms for 240 experts (60 layers x 4 experts x 9 comps)."""
    import fast_moe_load

    test_layers = 60  # full model

    fast_moe_load.init(
        num_workers=8,
        num_layers=test_layers,
        K=K,
        components=fml_components,
        packed_dir=PACKED_DIR,
        expert_size=expert_size,
    )

    # Generate random routing for all 60 layers
    rng = np.random.default_rng(42)
    routing = []
    for li in range(test_layers):
        experts = rng.choice(num_experts, size=K, replace=False).tolist()
        routing.append((li, experts))

    # Warmup (2 calls)
    fast_moe_load.load_and_assemble(routing)
    fast_moe_load.load_and_assemble(routing)

    # Reset stats after warmup
    # (can't reset individually, but total_calls gives us context)

    # Time the fresh-array approach with per-call timing
    N = 10
    timings = []
    for _ in range(N):
        s_before = fast_moe_load.stats()
        t0 = time.perf_counter()
        result = fast_moe_load.load_and_assemble(routing)
        t1 = time.perf_counter()
        s_after = fast_moe_load.stats()
        wall_ms = (t1 - t0) * 1000
        io_ms = s_after['total_io_ms'] - s_before['total_io_ms']
        create_ms = s_after['total_create_ms'] - s_before['total_create_ms']
        timings.append((wall_ms, io_ms, create_ms))

    s = fast_moe_load.stats()
    total_data_mb = sum(c['size'] for c in components) * K * test_layers / 1e6

    avg_wall = sum(t[0] for t in timings) / N
    avg_io = sum(t[1] for t in timings) / N
    avg_create = sum(t[2] for t in timings) / N
    min_wall = min(t[0] for t in timings)
    min_io = min(t[1] for t in timings)
    min_create = min(t[2] for t in timings)

    print(f"\n[TIMING] 60 layers, K={K}, {len(components)} components")
    print(f"  Data per token: {total_data_mb:.1f} MB ({K} experts x {test_layers} layers)")
    print(f"  Staging buffer: {s['staging_size']/(1024*1024):.1f} MB")
    print(f"  Avg wall:       {avg_wall:7.2f} ms")
    print(f"  Avg I/O:        {avg_io:7.2f} ms  (C pread, GIL released)")
    print(f"  Avg Create:     {avg_create:7.2f} ms  (numpy -> mx.array + eval)")
    print(f"  Min wall:       {min_wall:7.2f} ms")
    print(f"  Min I/O:        {min_io:7.2f} ms")
    print(f"  Min Create:     {min_create:7.2f} ms")
    print(f"  I/O throughput: {total_data_mb / (avg_io/1000):.0f} MB/s" if avg_io > 0 else "  I/O: N/A")
    print(f"\n  Per-call breakdown:")
    for i, (w, io, cr) in enumerate(timings):
        print(f"    Call {i}: wall={w:.1f}ms  io={io:.1f}ms  create={cr:.1f}ms")

    # Verify shape/dtype of one result
    assert len(result) == test_layers, f"Expected {test_layers} entries, got {len(result)}"
    first_dict = result[0]
    for comp in components:
        arr = first_dict[comp['name']]
        expected_shape = tuple([K] + comp['shape'])
        assert arr.shape == expected_shape, f"{comp['name']}: {arr.shape} != {expected_shape}"

    if t_total * 1000 < 50:
        print(f"\n  ** TARGET MET: {t_total*1000:.1f}ms < 50ms **")
    else:
        print(f"\n  Target: <50ms, actual: {t_total*1000:.1f}ms")

    fast_moe_load.shutdown()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing fast_moe_load C extension (FRESH mx.array approach)")
    print("=" * 60)
    print()

    all_pass = True
    for test in [test_init_and_stats, test_fresh_arrays, test_data_correctness, test_timing]:
        try:
            test()
            print()
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False
            print()

    if all_pass:
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
