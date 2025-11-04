#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_npy.py — 轻量级 .npy 文件检查脚本
------------------------------------------------
功能：
1) 检测并打印 .npy 文件的形状 (shape)、dtype、元素总数。
2) 打印一些示例值，避免一次性输出过大数据。
3) 支持目录 / 文件 / 通配符路径输入；对大数组使用内存映射读取（mmap）以节省内存。

用法：
    python inspect_npy.py data/*.npy
    python inspect_npy.py ./some_dir --max-examples 3
    python inspect_npy.py a.npy b.npy --mode random --seed 42
    python inspect_npy.py a.npy --per-dim 10 --show-stats

主要参数：
    paths           一个或多个文件/目录/通配符路径
    --max-examples  每个文件最多展示多少“示例”（默认 3）
    --mode          示例选择模式：auto/head/random/slices（默认 auto）
    --seed          随机示例的随机数种子（默认 0）
    --per-dim       除第 0 轴外，每个维度最多显示的元素数（默认 8）
    --show-stats    （可选）为数值型小数组计算 min/max/mean（可能较慢）

注意：
- 对于包含 Python 对象（object dtype）的 .npy，mmap 不可用，将退回普通加载。
- 示例输出做了尺寸限制，旨在“看个大概”。
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


STATS_MAX_ELEMS = 1_000_000  # 超过此元素数不计算统计量，避免过慢


def human_int(n: int) -> str:
    return f"{n:,}"


def find_npy_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        # 先按通配符展开
        expanded = list(map(Path, glob.glob(p))) if any(ch in p for ch in "*?[]") else [Path(p)]
        for path in expanded:
            if path.is_dir():
                files.extend(sorted(path.rglob("*.npy")))
            elif path.is_file() and path.suffix.lower() == ".npy":
                files.append(path)
            else:
                # 忽略不存在的路径或非 .npy 文件
                pass
    # 去重
    uniq = []
    seen = set()
    for f in files:
        try:
            key = f.resolve()
        except Exception:
            key = f
        if key not in seen:
            uniq.append(f)
            seen.add(key)
    return uniq


def safe_load_npy(path: Path):
    """
    优先使用 mmap 读取，若为 object dtype 或其他原因失败，则回退普通加载。
    返回：(array, used_mmap: bool)
    """
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        # 如果是 object dtype，mmap 不支持，改为普通加载（允许 pickle）
        if arr.dtype == object:
            raise ValueError("object dtype cannot be memory-mapped")
        return arr, True
    except Exception:
        # 回退：允许 pickle（可能会把文件完整加载进内存）
        arr = np.load(path, mmap_mode=None, allow_pickle=True)
        return arr, False


def example_indices(n: int, k: int) -> List[int]:
    if n <= 0 or k <= 0:
        return []
    # 固定选择：头部/中间/尾部，去重后最多 k 个
    idxs = [0, n // 2, n - 1]
    # 填足 k 个
    i = 1
    while len(idxs) < k and i < n - 1:
        if i not in idxs:
            idxs.append(i)
        i += 1
    # 去重并排序且确保在合法范围
    idxs = sorted({min(max(0, x), n - 1) for x in idxs})[:k]
    return idxs


def extract_examples(arr: np.ndarray, max_examples: int, mode: str, seed: int, per_dim: int) -> List[Tuple[str, np.ndarray]]:
    """
    返回一个列表 [(label, sample_array), ...]，用于打印。
    - 对 1D：返回 head 或随机下标的若干元素。
    - 对 2D：返回若干“行切片”。
    - 对 >=3D：固定第 0 轴取若干切片，并限制其他轴的长度为 per_dim。
    """
    rng = np.random.RandomState(seed)
    out: List[Tuple[str, np.ndarray]] = []

    if arr.ndim == 0:
        out.append(("scalar", arr[()]))
        return out

    if mode == "auto":
        mode = "head" if arr.ndim <= 1 else "slices"

    if arr.ndim == 1:
        n = arr.shape[0]
        if n == 0:
            return [("empty-1D", arr)]
        if mode == "random":
            k = min(max_examples, n)
            idxs = sorted(rng.choice(n, size=k, replace=False).tolist())
            out.append((f"1D random indices {idxs}", arr[idxs]))
        else:  # head
            k = min(max_examples * per_dim, n)  # 1D 适当多给点
            out.append((f"1D head[:{k}]", arr[:k]))
        return out

    if arr.ndim == 2:
        rows, cols = arr.shape
        if rows == 0 or cols == 0:
            return [("empty-2D", arr)]
        idxs = example_indices(rows, max_examples) if mode != "random" else sorted(rng.choice(rows, size=min(max_examples, rows), replace=False).tolist())
        col_lim = min(cols, per_dim)
        for r in idxs:
            sample = arr[r, :col_lim]
            out.append((f"row {r}, :{col_lim}", sample))
        return out

    # ndim >= 3
    axis0 = arr.shape[0]
    if axis0 == 0:
        return [("empty-nd", arr)]
    idxs = example_indices(axis0, max_examples) if mode != "random" else sorted(rng.choice(axis0, size=min(max_examples, axis0), replace=False).tolist())

    # 其他轴统一限制到 per_dim
    slicers = [slice(None)] * arr.ndim
    for i in range(1, arr.ndim):
        slicers[i] = slice(0, min(arr.shape[i], per_dim))

    for i0 in idxs:
        s = list(slicers)
        s[0] = i0
        sample = arr[tuple(s)]
        # 标注每个轴的裁剪情况
        cuts = ["*"] * arr.ndim
        cuts[0] = str(i0)
        for i in range(1, arr.ndim):
            cuts[i] = f":{min(arr.shape[i], per_dim)}"
        out.append((f"slice[{', '.join(cuts)}]", sample))
    return out


def maybe_numeric_stats(arr: np.ndarray) -> str:
    try:
        if arr.size > STATS_MAX_ELEMS:
            return f"(skip stats for large array > {human_int(STATS_MAX_ELEMS)} elems)"
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
            # 使用 nan 安全的聚合
            a = np.asanyarray(arr)
            mn = np.nanmin(a)
            mx = np.nanmax(a)
            mean = float(np.nanmean(a))
            return f"min={mn}, max={mx}, mean={mean:.6g}"
        return "(non-numeric dtype; stats skipped)"
    except Exception as e:
        return f"(stats error: {e})"


def main():
    parser = argparse.ArgumentParser(description="Inspect .npy files: show shape and example values.")
    parser.add_argument("paths", nargs="+", help="文件/目录/通配符路径（如 data/*.npy）")
    parser.add_argument("--max-examples", type=int, default=3, help="每个文件展示的示例数量上限（默认 3）")
    parser.add_argument("--mode", choices=["auto", "head", "random", "slices"], default="auto", help="示例选择模式（默认 auto）")
    parser.add_argument("--seed", type=int, default=0, help="随机模式的种子（默认 0）")
    parser.add_argument("--per-dim", type=int, default=8, help="除第 0 轴外，每个维度最多显示的元素数（默认 8）")
    parser.add_argument("--show-stats", action="store_true", help="为数值型小数组计算 min/max/mean（默认关闭）")

    args = parser.parse_args()

    np.set_printoptions(edgeitems=3, threshold=200, linewidth=120, suppress=True)

    files = find_npy_files(args.paths)
    if not files:
        print("未找到任何 .npy 文件。")
        return

    for idx, f in enumerate(files, 1):
        print("=" * 80)
        print(f"[{idx}/{len(files)}] {f}")
        try:
            arr, used_mmap = safe_load_npy(f)
            shape = arr.shape
            dtype = arr.dtype
            size = arr.size
            order = "F-contiguous" if arr.flags.f_contiguous else ("C-contiguous" if arr.flags.c_contiguous else "non-contiguous")
            mmap_info = "mmap=yes" if used_mmap else "mmap=no"

            print(f"shape={shape}, dtype={dtype}, size={human_int(size)}, {order}, {mmap_info}")

            if args.show_stats:
                print("stats:", maybe_numeric_stats(np.asarray(arr)))

            examples = extract_examples(np.asarray(arr), args.max_examples, args.mode, args.seed, args.per_dim)
            for label, sample in examples:
                print(f"- {label}:")
                print(sample)
        except KeyboardInterrupt:
            print("中断。")
            return
        except Exception as e:
            print(f"读取失败: {e}")

    print("=" * 80)
    print("完成。")


if __name__ == "__main__":
    main()
