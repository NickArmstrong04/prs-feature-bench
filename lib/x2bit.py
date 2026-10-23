#!/usr/bin/env python3
"""x2bit.py -- the 2-bit sample-major genotype store (X2.npy) and its decoder.

Layout: X2.npy is a uint8 array of shape (n_samples, ceil(n_snps / 4)); SNP
column j of sample i is bits 2*(j % 4) .. 2*(j % 4)+1 of byte X2[i, j // 4],
holding the dosage code 0/1/2 or 3 = missing (the same codes as the int8
X.npy of README s2). Pad SNPs beyond n_snps are 0. Each sample's genome is one
contiguous row, so a training batch of any sample subset is a row gather plus
unpack(); a block of SNP columns [c0, c1) with c0 % 4 == 0 is a byte slice.

    pack(block_int8)            (n, k) int8 codes 0..3 -> (n, ceil(k/4)) uint8
    unpack(packed, n_snps)      (n, B) uint8 -> (n, n_snps) int8
    batch(X2, rows, n_snps)     rows of a memmapped X2.npy -> (len(rows), n_snps) int8

Four shift-or passes over byte arrays; a 512 x 500k batch (64 MB packed)
unpacks in ~0.3 s on one core (measured locally), which DataLoader workers hide.
No dependency beyond numpy. This file is what the trainer's loader imports.
"""
import numpy as np

_SHIFT = np.array([0, 2, 4, 6], dtype=np.uint8)


def pack(block):
    """(n, k) int8/uint8 codes 0..3 -> (n, ceil(k/4)) uint8, little-end SNP first."""
    n, k = block.shape
    k4 = (k + 3) // 4 * 4
    if k4 != k:
        b = np.zeros((n, k4), dtype=np.uint8); b[:, :k] = block
    else:
        b = np.asarray(block, dtype=np.uint8) if block.dtype != np.uint8 else block
    out = (b[:, 0::4] | (b[:, 1::4] << 2)).astype(np.uint8)
    out |= b[:, 2::4] << 4
    out |= b[:, 3::4] << 6
    return out


def unpack(packed, n_snps):
    """(n, B) uint8 -> (n, n_snps) int8 codes 0..3 (3 = missing)."""
    packed = np.asarray(packed, dtype=np.uint8)
    n, B = packed.shape
    out = np.empty((n, B * 4), dtype=np.int8)
    for s in range(4):
        out[:, s::4] = (packed >> _SHIFT[s]) & 3
    return out[:, :n_snps]


def batch(X2, rows, n_snps):
    """Rows (any order, repeats allowed) of a memmapped X2.npy as (len(rows), n_snps) int8."""
    return unpack(X2[np.asarray(rows)], n_snps)


def open_x2(path, mode="r"):
    """np.load with mmap_mode; returns the (n, B) uint8 memmap."""
    return np.load(path, mmap_mode=mode)
