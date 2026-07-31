"""Binned / kernelized trace: continuous hits → discrete coverage.

``trace_measure`` is the discrete analogue of ``trace``: occupancy along a
ray after binning, with a 1D AA kernel. Leibniz boundary terms restore a
nonzero ``d(coverage)/d(split)``.

Fill and AA run on batched ``[B, K]`` / ``[B, W]`` arrays (no per-row Python).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp

import chalk.transform as tx
from chalk.trace import Trace, TraceTy, get_trace, trace_ray


def _fill_rows(splits, mask, n_bins: int):
    """Even-odd fill. ``splits, mask: [B, K]`` → coverage ``[B, n_bins]``."""
    splits = jnp.asarray(splits)
    mask = jnp.asarray(mask) > 0
    b, k = splits.shape
    split_int = jnp.floor(splits).astype(jnp.int32)
    valid = mask & (split_int >= 0) & (split_int < n_bins)
    loc = jnp.arange(k) % 2
    inout = jnp.where(loc == 0, 1.0, -1.0)
    ind = jnp.where(valid, split_int, n_bins)
    batch_ix = jnp.arange(b)[:, None]
    row = jnp.zeros((b, n_bins + 1))
    row = row.at[batch_ix, ind].add(jnp.where(valid, inout, 0.0))
    scene = jnp.cumsum(row[:, :-1], axis=-1)
    frac = splits - split_int
    edge = jnp.where(loc == 0, 1.0, 0.0) - inout * frac
    scene = scene.at[batch_ix, ind].set(jnp.where(valid, edge, 0.0), mode="drop")
    ok = jnp.mod(jnp.sum(mask.astype(jnp.int32), axis=-1), 2) == 0
    return jnp.where(ok[:, None], scene, jnp.zeros((b, n_bins)))


def _kernel(offset, kern: int):
    samples = jnp.arange(kern) - (kern // 2)
    k = kern - jnp.abs(samples - offset)
    denom = (kern - jnp.abs(samples)).sum()
    return jnp.maximum(0.0, k / denom)


def _convolve_rows(lines, kern: int):
    """``lines: [B, W]``."""
    if kern <= 1:
        return lines
    k = _kernel(0.0, kern)
    pad = kern // 2
    padded = jnp.pad(lines, ((0, 0), (pad, pad)))
    w = lines.shape[-1]
    idx = jnp.arange(w)[:, None] + jnp.arange(kern)[None, :]
    return (padded[:, idx] * k).sum(-1)


def _leibniz_row(g, f, splits, mask, kern: int):
    samples = jnp.arange(kern) - (kern // 2)
    n = f.shape[0]
    split_int = jnp.floor(splits).astype(jnp.int32)
    mask = jnp.asarray(mask) > 0

    def grad_p(s, s_off, m):
        off = s_off - s
        idx = s + samples
        ok = (idx >= 0) & (idx < n) & m
        idx_c = jnp.clip(idx, 0, n - 1)
        left = jnp.clip(s - 1, 0, n - 1)
        right = jnp.clip(s + 1, 0, n - 1)
        fm = f[right] - f[left]
        v = jnp.where(ok, g[idx_c], 0.0)
        return jnp.where(m, (v * fm) @ _kernel(off, kern), 0.0)

    r = jax.vmap(grad_p)(split_int, splits, mask)
    ok = jnp.mod(jnp.sum(mask.astype(jnp.int32)), 2) == 0
    return jnp.where(ok, r, jnp.zeros_like(splits))


@lru_cache(maxsize=8)
def _boundary_fn(kern: int):
    @jax.custom_vjp
    def boundary(coverage, splits, mask):
        return coverage

    def fwd(coverage, splits, mask):
        return coverage, (coverage, splits, mask)

    def bwd(res, g):
        f, splits, mask = res
        r = _leibniz_row(g, f, splits, mask, kern)
        return g, -r, None

    boundary.defvjp(fwd, bwd)
    return boundary


def measure_from_splits(
    splits, mask, n_bins: int, kernel: int = 11, *, boundary: bool = True
):
    """Coverage from precomputed splits. ``splits`` is ``[..., K]``."""
    splits = jnp.asarray(splits)
    mask = jnp.asarray(mask)
    squeeze = splits.ndim == 1
    if squeeze:
        splits = splits[None, ...]
        mask = jnp.reshape(jnp.broadcast_to(mask, splits.shape[-1]), (1, -1))
    else:
        batch = splits.shape[:-1]
        mask = jnp.broadcast_to(mask, splits.shape)
        splits = splits.reshape(-1, splits.shape[-1])
        mask = mask.reshape(-1, mask.shape[-1])
    cov = _convolve_rows(_fill_rows(splits, mask, n_bins), kernel)
    if boundary:
        bnd = _boundary_fn(int(kernel))
        cov = jax.vmap(bnd)(cov, splits, mask)
    if squeeze:
        return cov[0]
    return cov.reshape(batch + (n_bins,))


def trace_measure(
    obj: Any,
    p,
    v,
    n_bins: int,
    *,
    kernel: int = 11,
    pixel: float = 1.0,
    boundary: bool = True,
):
    """Discrete trace: occupancy along ray(s) after binning + AA.

    ``p + t v`` is partitioned into ``n_bins`` bins of width ``pixel``.
    Returns coverage ``α`` with shape ``broadcast(p,v).batch + (n_bins,)``.
    """
    is_trace = isinstance(obj, Trace) or isinstance(jax.typeof(obj), TraceTy)
    if not is_trace:
        obj = get_trace(obj)
    p = tx.data(p)
    v = tx.data(v)
    dists, mask = trace_ray(obj, p, v)
    splits = jnp.asarray(dists) / pixel
    return measure_from_splits(
        splits, mask, n_bins, kernel=kernel, boundary=boundary
    )


__all__ = ["trace_measure", "measure_from_splits"]
