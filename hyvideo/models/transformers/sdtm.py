import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _gather(input_tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    if input_tensor.device.type == "mps" and input_tensor.shape[-1] == 1:
        return torch.gather(
            input_tensor.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1),
        ).squeeze(-1)
    return torch.gather(input_tensor, dim, index)


def _init_generator(device: torch.device, fallback: Optional[torch.Generator] = None):
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    if device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    if fallback is not None:
        return fallback
    return _init_generator(torch.device("cpu"))


def build_sdtm_info(**kwargs) -> Dict[str, Any]:
    return {
        "type": "SDTM-HYVideo",
        "args": {
            "ratio": kwargs.get("ratio", 0.3),
            "deviation": kwargs.get("deviation", 0.2),
            "switch_step": kwargs.get("switch_step", 20),
            "use_rand": kwargs.get("use_rand", True),
            "sx": kwargs.get("sx", 4),
            "sy": kwargs.get("sy", 4),
            "a_s": kwargs.get("a_s", 0.05),
            "adaptive_ssm": kwargs.get("adaptive_ssm", False),
            "ssm_threshold": kwargs.get("ssm_threshold", 0.0),
            "low_complexity_threshold": kwargs.get("low_complexity_threshold", 0.66),
            "high_complexity_threshold": kwargs.get("high_complexity_threshold", 0.33),
            "pseudo_merge": kwargs.get("pseudo_merge", False),
            "mcw": kwargs.get("mcw", 0.2),
            "protect_steps_frequency": kwargs.get("protect_steps_frequency", 3),
            "protect_layers_frequency": kwargs.get("protect_layers_frequency", -1),
            "cache_each_step": kwargs.get("cache_each_step", True),
            "merge_attn": kwargs.get("merge_attn", True),
            "merge_mlp": kwargs.get("merge_mlp", True),
            "auto_window": kwargs.get("auto_window", True),
            "slic_alpha": kwargs.get("slic_alpha", 0.5),
            "generator": None,
            "verbose": kwargs.get("verbose", False),
        },
        "features": {
            "attn_output": {},
            "mlp_output": {},
        },
        "states": {
            "enabled": True,
            "step_count": None,
            "step_current": None,
            "chunk_current": None,
            "layer_count": None,
            "layer_current": None,
            "ratio_current": kwargs.get("ratio", 0.3),
            "last_independent": None,
            "last_independent_shape": None,
            "last_window": None,
            "skip_warning_printed": set(),
        },
    }


def begin_sdtm_run(module: torch.nn.Module, step_count: int):
    info = getattr(module, "_sdtm_info", None)
    if not info:
        return
    info["states"]["step_count"] = int(step_count)
    info["states"]["step_current"] = None
    info["states"]["chunk_current"] = None
    info["states"]["last_independent"] = None
    info["states"]["last_independent_shape"] = None
    info["features"] = {"attn_output": {}, "mlp_output": {}}


def set_sdtm_step(
    module: torch.nn.Module,
    step_current: int,
    step_count: Optional[int] = None,
    chunk_current: Optional[int] = None,
):
    info = getattr(module, "_sdtm_info", None)
    if not info:
        return
    if step_count is not None:
        info["states"]["step_count"] = int(step_count)
    info["states"]["step_current"] = int(step_current)
    info["states"]["chunk_current"] = None if chunk_current is None else int(chunk_current)


def disable_sdtm(module: torch.nn.Module):
    if hasattr(module, "_sdtm_info"):
        module._sdtm_info["states"]["enabled"] = False


def store_sdtm_feature(
    info: Optional[Dict[str, Any]],
    phase: str,
    layer_idx: Optional[int],
    tensor: torch.Tensor,
):
    if not info or not info.get("states", {}).get("enabled", False):
        return
    if not info.get("args", {}).get("cache_each_step", True):
        return
    if phase not in info.get("features", {}):
        return
    if layer_idx is None:
        layer_idx = info.get("states", {}).get("layer_current", -1)
    info["features"][phase][f"l{layer_idx}"] = tensor.detach()


def _is_protected(idx: int, total: int, freq: Optional[int]) -> bool:
    if freq is None or freq < 0 or freq == 0:
        return False
    if idx % freq == 0:
        return True
    return idx == max(total - 1, 0)


def compute_ratio(info: Dict[str, Any]) -> float:
    args = info.get("args", {})
    states = info.get("states", {})
    ratio = float(args.get("ratio", 0.3))
    deviation = float(args.get("deviation", 0.2))
    step_current = states.get("step_current")
    step_count = states.get("step_count") or 1
    layer_current = states.get("layer_current") or 0
    layer_count = states.get("layer_count") or 1

    if step_current is None:
        return 0.0
    if _is_protected(
        layer_current, layer_count, args.get("protect_layers_frequency", -1)
    ):
        states["last_independent"] = None
        states["last_independent_shape"] = None
        return 0.0

    progress = step_current / max(step_count - 1, 1)
    alpha = math.cos(progress * math.pi / 2)
    return float((ratio - deviation) + (2.0 * deviation) * alpha)


def _best_divisor(value: int, preferred: int) -> int:
    preferred = max(int(preferred), 1)
    for candidate in range(min(preferred, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _resolve_window(
    h: int, w: int, sy: int, sx: int, auto_window: bool
) -> Optional[Tuple[int, int]]:
    sy = max(int(sy), 1)
    sx = max(int(sx), 1)
    if h % sy == 0 and w % sx == 0 and sy * sx >= 4:
        return sy, sx
    if not auto_window:
        return None

    sy = _best_divisor(h, sy)
    sx = _best_divisor(w, sx)
    if sy * sx >= 4:
        return sy, sx

    best = None
    best_area = 0
    for dh in range(1, h + 1):
        if h % dh != 0:
            continue
        for dw in range(1, w + 1):
            if w % dw != 0:
                continue
            area = dh * dw
            if 4 <= area and area > best_area:
                best = (dh, dw)
                best_area = area
    return best


def _update_last_independent(
    info: Dict[str, Any], independent_indices: torch.Tensor, total_tokens: int
):
    states = info["states"]
    shape = (independent_indices.shape[0], total_tokens)
    last_ind = states.get("last_independent")
    if last_ind is None or tuple(last_ind.shape) != shape:
        last_ind = torch.zeros(
            shape,
            device=independent_indices.device,
            dtype=torch.int32,
        )
        states["last_independent"] = last_ind
        states["last_independent_shape"] = shape

    last_ind.add_(1)
    zeros = torch.zeros_like(independent_indices, dtype=last_ind.dtype)
    last_ind.scatter_(1, independent_indices, zeros)


@dataclass
class SDTMMergeContext:
    info: Optional[Dict[str, Any]]
    t: int = 0
    h: int = 0
    w: int = 0
    unm_idx: Optional[torch.Tensor] = None
    src_idx: Optional[torch.Tensor] = None
    dst_idx: Optional[torch.Tensor] = None
    merge_idx: Optional[torch.Tensor] = None
    attn_enabled: bool = False
    mlp_enabled: bool = False

    @property
    def active(self) -> bool:
        return self.unm_idx is not None and (self.attn_enabled or self.mlp_enabled)

    @property
    def tokens_per_frame(self) -> int:
        return self.h * self.w

    @property
    def reduced_tokens_per_frame(self) -> int:
        if self.unm_idx is None or self.dst_idx is None:
            return self.tokens_per_frame
        return int(self.unm_idx.shape[1] + self.dst_idx.shape[1])

    def _idx(self, idx: torch.Tensor, batch: int, channels: int) -> torch.Tensor:
        return idx.unsqueeze(0).expand(batch, self.t, idx.shape[1], channels)

    def _flatten_feature(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        trailing = tuple(x.shape[2:])
        flat = x.reshape(x.shape[0], self.t, self.tokens_per_frame, -1)
        return flat, trailing

    def _restore_feature(
        self, x: torch.Tensor, batch: int, tokens: int, trailing: Tuple[int, ...]
    ) -> torch.Tensor:
        return x.reshape(batch, self.t * tokens, *trailing)

    def _merge_impl(self, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if not self.active:
            return x
        batch = x.shape[0]
        flat, trailing = self._flatten_feature(x)
        channels = flat.shape[-1]

        unm = _gather(flat, 2, self._idx(self.unm_idx, batch, channels))
        dst = _gather(flat, 2, self._idx(self.dst_idx, batch, channels))
        src_len = self.src_idx.shape[1]
        if src_len > 0:
            src = _gather(flat, 2, self._idx(self.src_idx, batch, channels))
            dst = dst.scatter_reduce(
                2,
                self._idx(self.merge_idx, batch, channels),
                src,
                reduce=mode,
            )
        merged = torch.cat([unm, dst], dim=2)
        return self._restore_feature(
            merged, batch, self.reduced_tokens_per_frame, trailing
        )

    def _prune_impl(self, x: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return x
        batch = x.shape[0]
        flat, trailing = self._flatten_feature(x)
        channels = flat.shape[-1]
        unm = _gather(flat, 2, self._idx(self.unm_idx, batch, channels))
        dst = _gather(flat, 2, self._idx(self.dst_idx, batch, channels))
        pruned = torch.cat([unm, dst], dim=2)
        return self._restore_feature(
            pruned, batch, self.reduced_tokens_per_frame, trailing
        )

    def _unmerge_impl(self, x: torch.Tensor, phase: str) -> torch.Tensor:
        if not self.active:
            return x
        batch = x.shape[0]
        trailing = tuple(x.shape[2:])
        channels = int(torch.tensor(trailing).prod().item()) if trailing else 1
        flat = x.reshape(batch, self.t, self.reduced_tokens_per_frame, channels)

        unm_len = self.unm_idx.shape[1]
        dst_len = self.dst_idx.shape[1]
        src_len = self.src_idx.shape[1]
        unm = flat[:, :, :unm_len, :]
        dst = flat[:, :, unm_len : unm_len + dst_len, :]
        if src_len > 0:
            src = _gather(dst, 2, self._idx(self.merge_idx, batch, channels))
        else:
            src = dst[:, :, :0, :]

        cache_full = None
        if self.info is not None:
            layer = self.info.get("states", {}).get("layer_current", -1)
            cache_full = (
                self.info.get("features", {})
                .get(phase, {})
                .get(f"l{layer}", None)
            )
        if cache_full is not None and cache_full.shape[1] == self.t * self.tokens_per_frame:
            try:
                mcw = float(self.info.get("args", {}).get("mcw", 1.0))
                cache_flat = cache_full.to(device=x.device, dtype=x.dtype).reshape(
                    batch, self.t, self.tokens_per_frame, channels
                )
                cached_dst = _gather(
                    cache_flat, 2, self._idx(self.dst_idx, batch, channels)
                )
                cached_src = _gather(
                    cache_flat, 2, self._idx(self.src_idx, batch, channels)
                )
                dst = mcw * dst + (1.0 - mcw) * cached_dst
                src = mcw * src + (1.0 - mcw) * cached_src
            except Exception:
                pass

        out = torch.zeros(
            batch,
            self.t,
            self.tokens_per_frame,
            channels,
            device=x.device,
            dtype=x.dtype,
        )
        out.scatter_(2, self._idx(self.unm_idx, batch, channels), unm)
        out.scatter_(2, self._idx(self.dst_idx, batch, channels), dst)
        if src_len > 0:
            out.scatter_(2, self._idx(self.src_idx, batch, channels), src)
        return out.reshape(batch, self.t * self.tokens_per_frame, *trailing)

    def merge_attn(self, x: torch.Tensor) -> torch.Tensor:
        return self._merge_impl(x) if self.attn_enabled else x

    def merge_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self._merge_impl(x) if self.mlp_enabled else x

    def unmerge_attn(self, x: torch.Tensor, phase: str = "attn_output") -> torch.Tensor:
        return self._unmerge_impl(x, phase) if self.attn_enabled else x

    def unmerge_mlp(self, x: torch.Tensor, phase: str = "mlp_output") -> torch.Tensor:
        return self._unmerge_impl(x, phase) if self.mlp_enabled else x

    def prune_attn_like(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or not self.attn_enabled:
            return x
        if x.ndim >= 3 and x.shape[1] == self.t * self.tokens_per_frame:
            return self._prune_impl(x)
        if x.ndim >= 2 and x.shape[0] == self.t * self.tokens_per_frame:
            trailing = tuple(x.shape[1:])
            flat = x.reshape(self.t, self.tokens_per_frame, -1)
            channels = flat.shape[-1]
            idx = torch.cat([self.unm_idx, self.dst_idx], dim=1).expand(
                self.t, self.reduced_tokens_per_frame, channels
            )
            out = _gather(flat, 1, idx)
            return out.reshape(self.t * self.reduced_tokens_per_frame, *trailing)
        return x

    def prune_attn_freqs(self, freqs_cis):
        if freqs_cis is None or not self.attn_enabled:
            return freqs_cis
        return tuple(self.prune_attn_like(freq) for freq in freqs_cis)


def _ssm_indices(
    metric: torch.Tensor,
    h: int,
    w: int,
    reduce_num: Optional[int],
    threshold: float,
    window_size: Tuple[int, int],
    no_rand: bool,
    generator: Optional[torch.Generator],
    info: Dict[str, Any],
):
    groups, tokens, channels = metric.shape
    ws_h, ws_w = int(window_size[0]), int(window_size[1])
    stride_h, stride_w = ws_h, ws_w
    k_tokens = ws_h * ws_w
    if k_tokens < 4 or h % ws_h != 0 or w % ws_w != 0:
        return None

    metric_grid = metric.view(groups, h, w, channels).permute(0, 3, 1, 2)
    metric_windows = metric_grid.view(
        groups, channels, h // ws_h, ws_h, w // ws_w, ws_w
    ).permute(0, 2, 4, 1, 3, 5)
    _, gh, gw, c, _, _ = metric_windows.shape

    tensor_flattened = metric_windows.reshape(groups, gh, gw, c, -1)
    window_feat_for_sim = tensor_flattened.permute(0, 1, 2, 4, 3)
    window_feat_for_sim = F.normalize(window_feat_for_sim, dim=-1, eps=1e-6)
    sims = window_feat_for_sim @ window_feat_for_sim.transpose(-1, -2)
    similarity_map = sims.sum(-1).sum(-1) / (k_tokens * k_tokens)
    variance_raw = sims.var(dim=(-1, -2), unbiased=False)
    similarity_map = similarity_map.reshape(groups, -1)
    variance_raw = variance_raw.reshape(groups, -1)

    v_min = variance_raw.amin(dim=1, keepdim=True)
    v_max = variance_raw.amax(dim=1, keepdim=True)
    variance_map = 1.0 - (variance_raw - v_min) / (v_max - v_min + 1e-6)

    indiv_priority_flat = torch.zeros_like(similarity_map)
    last_ind = info.get("states", {}).get("last_independent")
    if last_ind is not None and tuple(last_ind.shape) == (groups, tokens):
        li_f = last_ind.to(similarity_map.dtype)
        indiv_priority = li_f / (li_f.mean(dim=1, keepdim=True) + 1e-6)
        indiv_priority_windows = (
            indiv_priority.view(groups, h, w)
            .view(groups, gh, ws_h, gw, ws_w)
            .permute(0, 1, 3, 2, 4)
            .reshape(groups, gh, gw, k_tokens)
            .mean(-1)
        )
        indiv_priority_flat = indiv_priority_windows.view(groups, gh * gw)

    a_s = float(info.get("args", {}).get("a_s", 0.0))
    score_map = similarity_map + variance_map + a_s * indiv_priority_flat

    token_grid = torch.arange(tokens, device=metric.device).reshape(h, w)
    token_grid = token_grid.unsqueeze(0).repeat(groups, 1, 1)
    windowed_tokens = token_grid.unfold(1, ws_h, stride_h).unfold(2, ws_w, stride_w)
    windowed_tokens = windowed_tokens.reshape(groups, -1, k_tokens)
    num_windows = windowed_tokens.shape[1]

    selected_class = torch.full(
        (groups, num_windows), -1, dtype=torch.long, device=metric.device
    )
    low_thr = float(info.get("args", {}).get("low_complexity_threshold", 0.66))
    high_thr = float(info.get("args", {}).get("high_complexity_threshold", 0.33))
    low_thr, high_thr = max(low_thr, high_thr), min(low_thr, high_thr)

    if reduce_num is None:
        cand_mask = score_map >= float(threshold)
        class_masks = [
            cand_mask & (variance_map >= low_thr),
            cand_mask & (variance_map < low_thr) & (variance_map >= high_thr),
            cand_mask & (variance_map < high_thr),
        ]
        for cls_id, cls_mask in enumerate(class_masks):
            keep_count = int(cls_mask.sum(dim=1).min().item())
            if keep_count > 0:
                chosen = score_map.masked_fill(~cls_mask, -1e9).topk(
                    keep_count, dim=-1
                ).indices
                selected_class.scatter_(
                    1,
                    chosen,
                    torch.full(
                        (groups, keep_count),
                        cls_id,
                        dtype=torch.long,
                        device=metric.device,
                    ),
                )
    else:
        selected_count = min(int(reduce_num), num_windows)
        if selected_count <= 0:
            return None
        chosen = score_map.topk(selected_count, dim=-1).indices
        selected_var = variance_map.gather(1, chosen)
        order = selected_var.argsort(dim=-1, descending=True)
        sorted_chosen = chosen.gather(1, order)
        low_n = int(selected_count * 0.25)
        mid_n = int(selected_count * 0.25)
        if low_n > 0:
            selected_class.scatter_(
                1,
                sorted_chosen[:, :low_n],
                torch.zeros(groups, low_n, dtype=torch.long, device=metric.device),
            )
        if mid_n > 0:
            selected_class.scatter_(
                1,
                sorted_chosen[:, low_n : low_n + mid_n],
                torch.ones(groups, mid_n, dtype=torch.long, device=metric.device),
            )
        high_n = selected_count - low_n - mid_n
        if high_n > 0:
            selected_class.scatter_(
                1,
                sorted_chosen[:, low_n + mid_n :],
                torch.full((groups, high_n), 2, dtype=torch.long, device=metric.device),
            )

    window_feat = tensor_flattened.permute(0, 1, 2, 4, 3).reshape(
        groups, num_windows, k_tokens, c
    )
    wf_norm = window_feat / (window_feat.norm(dim=-1, keepdim=True) + 1e-6)
    if no_rand:
        rand_order = torch.arange(k_tokens, device=metric.device).view(1, 1, k_tokens)
        rand_order = rand_order.expand(groups, num_windows, k_tokens)
    else:
        rand_noise = torch.rand(
            groups, num_windows, k_tokens, device=metric.device, generator=generator
        )
        rand_order = rand_noise.argsort(dim=-1)
    shuffled_tokens = windowed_tokens.gather(2, rand_order)

    all_src, all_dst, all_merge = [], [], []
    dst_cum = 0
    for cls_id, keep_n in [(0, 1), (1, 2), (2, min(4, k_tokens - 1))]:
        if keep_n <= 0 or keep_n >= k_tokens:
            continue
        src_n = k_tokens - keep_n
        cls_mask = selected_class == cls_id
        n_cls = int(cls_mask.sum(dim=1).min().item())
        if n_cls == 0:
            continue
        cls_win_idx = cls_mask.float().topk(n_cls, dim=-1).indices
        tokens_cls = shuffled_tokens.gather(
            1, cls_win_idx.unsqueeze(-1).expand(groups, n_cls, k_tokens)
        )
        dst_tokens = tokens_cls[:, :, :keep_n]
        src_tokens = tokens_cls[:, :, keep_n:]
        if keep_n == 1:
            merge_local = torch.zeros(
                groups, n_cls, src_n, dtype=torch.long, device=metric.device
            )
        else:
            wf_cls = wf_norm.gather(
                1,
                cls_win_idx.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(groups, n_cls, k_tokens, c),
            )
            rand_cls = rand_order.gather(
                1, cls_win_idx.unsqueeze(-1).expand(groups, n_cls, k_tokens)
            )
            wf_cls = wf_cls.gather(
                2, rand_cls.unsqueeze(-1).expand(groups, n_cls, k_tokens, c)
            )
            src_feat = wf_cls[:, :, keep_n:, :]
            dst_feat = wf_cls[:, :, :keep_n, :]
            merge_local = torch.einsum("gwsc,gwdc->gwsd", src_feat, dst_feat).argmax(
                dim=-1
            )

        win_base = torch.arange(n_cls, device=metric.device).view(1, n_cls, 1) * keep_n
        merge_global = dst_cum + win_base + merge_local
        all_dst.append(dst_tokens.reshape(groups, n_cls * keep_n).unsqueeze(-1))
        all_src.append(src_tokens.reshape(groups, n_cls * src_n).unsqueeze(-1))
        all_merge.append(merge_global.reshape(groups, n_cls * src_n).unsqueeze(-1))
        dst_cum += n_cls * keep_n

    unm_mask = selected_class == -1
    n_unm = int(unm_mask.sum(dim=1).min().item())
    if n_unm > 0:
        unm_win_idx = unm_mask.float().topk(n_unm, dim=-1).indices
        unm_tokens = windowed_tokens.gather(
            1, unm_win_idx.unsqueeze(-1).expand(groups, n_unm, k_tokens)
        )
        unm_idx = unm_tokens.reshape(groups, n_unm * k_tokens).unsqueeze(-1)
    else:
        unm_idx = torch.zeros(groups, 0, 1, device=metric.device, dtype=torch.long)

    src_idx = (
        torch.cat(all_src, dim=1)
        if all_src
        else torch.zeros(groups, 0, 1, device=metric.device, dtype=torch.long)
    )
    dst_idx = (
        torch.cat(all_dst, dim=1)
        if all_dst
        else torch.zeros(groups, 0, 1, device=metric.device, dtype=torch.long)
    )
    merge_idx = (
        torch.cat(all_merge, dim=1)
        if all_merge
        else torch.zeros(groups, 0, 1, device=metric.device, dtype=torch.long)
    )

    if info.get("args", {}).get("pseudo_merge", False):
        independent_idx = torch.cat([unm_idx.squeeze(-1), dst_idx.squeeze(-1)], dim=-1)
    else:
        independent_idx = unm_idx.squeeze(-1)
    return unm_idx, src_idx, dst_idx, merge_idx, independent_idx


def _fidm_indices(
    metric: torch.Tensor,
    h: int,
    w: int,
    reduce_num: int,
    window_size: Tuple[int, int],
    no_rand: bool,
    generator: Optional[torch.Generator],
    info: Dict[str, Any],
):
    groups, tokens, channels = metric.shape
    sy, sx = int(window_size[0]), int(window_size[1])
    if h % sy != 0 or w % sx != 0 or sy * sx < 2:
        return None
    hsy, wsx = h // sy, w // sx
    num_dst = hsy * wsx

    last_ind = info.get("states", {}).get("last_independent")
    if last_ind is not None and tuple(last_ind.shape) == (groups, tokens):
        li_windows = (
            last_ind.view(groups, h, w)
            .view(groups, hsy, sy, wsx, sx)
            .permute(0, 1, 3, 2, 4)
            .reshape(groups, hsy, wsx, sy * sx)
        )
        dst_pos = li_windows.argmax(dim=-1, keepdim=True)
    elif no_rand:
        dst_pos = torch.zeros(
            groups, hsy, wsx, 1, device=metric.device, dtype=torch.long
        )
    else:
        dst_pos = torch.randint(
            sy * sx,
            (groups, hsy, wsx, 1),
            device=metric.device,
            generator=generator,
        )

    idx_buffer_view = torch.zeros(
        groups, hsy, wsx, sy * sx, device=metric.device, dtype=torch.long
    )
    idx_buffer_view.scatter_(3, dst_pos, -torch.ones_like(dst_pos))
    idx_buffer = (
        idx_buffer_view.view(groups, hsy, wsx, sy, sx)
        .transpose(2, 3)
        .reshape(groups, h, w)
    )
    rand_idx = idx_buffer.reshape(groups, -1, 1).argsort(dim=1)
    a_idx = rand_idx[:, num_dst:, :]
    b_idx = rand_idx[:, :num_dst, :]

    def split(x):
        c = x.shape[-1]
        src = _gather(x, 1, a_idx.expand(groups, tokens - num_dst, c))
        dst = _gather(x, 1, b_idx.expand(groups, num_dst, c))
        return src, dst

    metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
    src_metric, dst_metric = split(metric)
    feature_sim = src_metric @ dst_metric.transpose(-1, -2)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=metric.device),
        torch.arange(w, device=metric.device),
        indexing="ij",
    )
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(tokens, 2).float()
    coords[:, 0] = coords[:, 0] / max(w - 1, 1)
    coords[:, 1] = coords[:, 1] / max(h - 1, 1)
    coords = coords.unsqueeze(0).expand(groups, tokens, 2)
    coord_src = _gather(coords, 1, a_idx.expand(groups, tokens - num_dst, 2))
    coord_dst = _gather(coords, 1, b_idx.expand(groups, num_dst, 2))
    spatial_dist = torch.cdist(coord_src, coord_dst, p=2)
    alpha = float(info.get("args", {}).get("slic_alpha", 0.5))
    scores = feature_sim - alpha * spatial_dist

    reduce_num = min(tokens - num_dst, int(reduce_num))
    reduce_num = reduce_num // 16 * 16
    if reduce_num <= 0:
        return None

    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx_rel = edge_idx[..., reduce_num:, :]
    src_idx_rel = edge_idx[..., :reduce_num, :]
    merge_idx = _gather(node_idx[..., None], 1, src_idx_rel)

    dst_idx = b_idx
    unm_idx = _gather(a_idx, 1, unm_idx_rel)
    src_idx = _gather(a_idx, 1, src_idx_rel)

    if info.get("args", {}).get("pseudo_merge", False):
        independent_idx = torch.cat([unm_idx.squeeze(-1), dst_idx.squeeze(-1)], dim=-1)
    else:
        independent_idx = unm_idx.squeeze(-1)
    return unm_idx, src_idx, dst_idx, merge_idx, independent_idx


def get_sdtm_context(
    x: torch.Tensor,
    info: Optional[Dict[str, Any]],
    attn_param: Optional[Dict[str, Any]],
    block_idx: Optional[int],
    cache_vision: bool = False,
) -> SDTMMergeContext:
    if (
        info is None
        or not info.get("states", {}).get("enabled", False)
        or cache_vision
        or attn_param is None
        or "thw" not in attn_param
    ):
        return SDTMMergeContext(info)

    states = info["states"]
    args = info["args"]
    states["layer_current"] = int(block_idx or 0)
    if states.get("layer_count") is None:
        states["layer_count"] = int(block_idx or 0) + 1

    thw = attn_param["thw"]
    t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
    if x.shape[1] != t * h * w:
        key = f"shape-{x.shape[1]}-{t}-{h}-{w}"
        if args.get("verbose", False) and key not in states["skip_warning_printed"]:
            print(
                f"[SDTM] skip block {block_idx}: local token length {x.shape[1]} "
                f"does not match full THW {t}x{h}x{w}.",
                flush=True,
            )
            states["skip_warning_printed"].add(key)
        return SDTMMergeContext(info)

    step_current = states.get("step_current")
    step_count = states.get("step_count") or 1
    if step_current is None:
        return SDTMMergeContext(info)

    if _is_protected(
        int(step_current), int(step_count), args.get("protect_steps_frequency", 3)
    ):
        return SDTMMergeContext(info)

    window = _resolve_window(
        h,
        w,
        args.get("sy", 4),
        args.get("sx", 4),
        bool(args.get("auto_window", True)),
    )
    if window is None:
        return SDTMMergeContext(info)
    if args.get("verbose", False) and states.get("last_window") != window:
        print(f"[SDTM] using per-frame window {window[1]}x{window[0]}", flush=True)
        states["last_window"] = window

    ratio_current = compute_ratio(info)
    states["ratio_current"] = ratio_current
    reduce_num = int(h * w * ratio_current)
    if reduce_num <= 0:
        return SDTMMergeContext(info)

    if args["generator"] is None or args["generator"].device != x.device:
        args["generator"] = _init_generator(x.device, fallback=args["generator"])

    metric = x.detach().reshape(x.shape[0], t, h * w, x.shape[-1]).mean(dim=0)
    metric = metric.float()
    no_rand = not bool(args.get("use_rand", True))
    if int(step_current) <= int(args.get("switch_step", 20)):
        indices = _ssm_indices(
            metric,
            h,
            w,
            None if args.get("adaptive_ssm", False) else reduce_num,
            float(args.get("ssm_threshold", 0.0)),
            window,
            no_rand,
            args["generator"],
            info,
        )
    else:
        indices = _fidm_indices(
            metric,
            h,
            w,
            reduce_num,
            window,
            no_rand,
            args["generator"],
            info,
        )
    if indices is None:
        return SDTMMergeContext(info)

    unm_idx, src_idx, dst_idx, merge_idx, independent_idx = indices
    _update_last_independent(info, independent_idx, h * w)

    early_step = int(step_current) <= int(args.get("switch_step", 20))
    attn_enabled = bool(args.get("merge_attn", True)) and early_step
    mlp_enabled = bool(args.get("merge_mlp", True))
    return SDTMMergeContext(
        info=info,
        t=t,
        h=h,
        w=w,
        unm_idx=unm_idx,
        src_idx=src_idx,
        dst_idx=dst_idx,
        merge_idx=merge_idx,
        attn_enabled=attn_enabled,
        mlp_enabled=mlp_enabled,
    )
