#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from datasets import disable_progress_bar, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


# =============================================================================
# Environment
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAMB_ROOT = os.environ.get(
    "LAMB_ROOT", str(PROJECT_ROOT / "third_party/lamb")
)

sys.path.insert(0, LAMB_ROOT)

from mamba_ssm.mapping import load_dt_remapping_from_config
from mamba_ssm.mapping.remap_simple import DtRemappingBase
from mamba_ssm.mapping.utils import dt_bias_activate
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from tasks.longbench_utils import (
    dataset2maxgenlen,
    dataset2metric,
    dataset2prompt,
)


MODEL_NAME = "state-spaces/mamba2-1.3b"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"

DEFAULT_CONFIG = (
    "remapping_configs/"
    "mamba2-1.3b_topk1024_pk9.json"
)

ARMS = [
    "original_lamb",
    "canonical_full_shared",
    "bounded_local_shared",
    "bounded_local_band",
    "fusion_all",
]

DEFAULT_DATASETS = [
    "qasper",
    "hotpotqa",
    "gov_report",
    "passage_retrieval_en",
    "lcc",
    "samsum",
]


# =============================================================================
# Generic utilities
# =============================================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(tmp, path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    path = Path(path)

    if not path.exists():
        return []

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def finite_mean(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return float("nan")

    return float(x.mean())


def jaccard(a, b):
    a = a.bool()
    b = b.bool()

    inter = (
        a & b
    ).sum().item()

    union = (
        a | b
    ).sum().item()

    if union == 0:
        return 1.0

    return float(
        inter / union
    )


# =============================================================================
# LongBench metric
# =============================================================================

def score_prediction(
    dataset,
    prediction,
    answers,
    all_classes,
):
    pred = prediction

    if dataset in [
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
    ]:
        pred = (
            pred
            .lstrip("\n")
            .split("\n")[0]
        )

    best = 0.0

    for gt in answers:
        value = dataset2metric[
            dataset
        ](
            pred,
            gt,
            all_classes=all_classes,
        )

        best = max(
            best,
            float(value),
        )

    return best * 100.0


# =============================================================================
# Canonical rolling mean/std normalization
# =============================================================================

def rolling_mean_std_zscore(
    x,
    window,
    eps=1e-6,
):
    """
    Canonical local z-score.

    IMPORTANT:
      - mean/std, not median/MAD
      - no positive clipping
      - no quantile transform
      - negative scores remain negative
    """

    x = x.float()

    n = int(
        x.numel()
    )

    if n <= 1:
        return torch.zeros_like(x)

    window = min(
        int(window),
        (
            n
            if n % 2 == 1
            else n - 1
        ),
    )

    window = max(
        1,
        window,
    )

    if window % 2 == 0:
        window -= 1

    if window <= 1:
        mean = x.mean()
        std = x.std(
            unbiased=False
        )

        return (
            x - mean
        ) / (
            std + eps
        )

    pad = window // 2

    padded = F.pad(
        x[
            None,
            None,
            :
        ],
        (
            pad,
            pad,
        ),
        mode="replicate",
    )[
        0,
        0
    ]

    windows = padded.unfold(
        0,
        window,
        1,
    )

    mean = windows.mean(
        dim=1
    )

    std = windows.std(
        dim=1,
        unbiased=False,
    )

    return (
        x - mean
    ) / (
        std + eps
    )


# =============================================================================
# Exact official LAMB pooling / budget
# =============================================================================

def official_mean_pool(
    score,
    kernel,
):
    """
    Exact form used by official LAMB:

      avg_pool1d(
          score[None,None],
          kernel,
          stride=1,
          padding=kernel//2
      )

    For even kernel=18 this has length input+1,
    which matches the official implementation.
    """

    return F.avg_pool1d(
        score[
            None,
            None,
            :
        ],
        kernel_size=int(kernel),
        stride=1,
        padding=int(kernel) // 2,
    )[
        0,
        0
    ]


def build_official_budget_mask_from_pooled(
    pooled_score,
    seq_len,
    topk,
    vis_window,
    device,
):
    """
    Match official LAMB TopK semantics.

    If sequence is shorter than topk + vis_window,
    official LAMB does no filtering.
    """

    mask = torch.zeros(
        seq_len,
        dtype=torch.bool,
        device=device,
    )

    if (
        seq_len
        <
        int(topk)
        +
        int(vis_window)
    ):
        mask[:] = True
        return mask

    k = min(
        int(topk),
        int(
            pooled_score.numel()
        ),
    )

    top_idx = torch.topk(
        pooled_score,
        k=k,
        largest=True,
        sorted=False,
    ).indices

    # pooled index corresponds directly to sequence position
    mask[
        top_idx
    ] = True

    if vis_window > 0:
        mask[
            -int(vis_window):
        ] = True

    return mask


def build_mask_from_position_scores(
    position_score,
    seq_len,
    topk,
    vis_window,
    pooling_kernel,
):
    if (
        seq_len
        <
        int(topk)
        +
        int(vis_window)
    ):
        return torch.ones(
            seq_len,
            dtype=torch.bool,
            device=position_score.device,
        ), None

    history_end = (
        seq_len
        -
        int(vis_window)
    )

    history = position_score[
        :history_end
    ]

    pooled = official_mean_pool(
        history,
        pooling_kernel,
    )

    mask = (
        build_official_budget_mask_from_pooled(
            pooled_score=pooled,
            seq_len=seq_len,
            topk=topk,
            vis_window=vis_window,
            device=position_score.device,
        )
    )

    return (
        mask,
        pooled,
    )


# =============================================================================
# Standard full-context causal Surprisal
# =============================================================================

def compute_final_surprisal(
    model,
    hidden,
    input_ids,
    chunk_size,
):
    seq_len = int(
        input_ids.shape[1]
    )

    result = torch.zeros(
        seq_len,
        dtype=torch.float32,
        device=input_ids.device,
    )

    for start in range(
        0,
        seq_len - 1,
        int(chunk_size),
    ):
        end = min(
            start + int(chunk_size),
            seq_len - 1,
        )

        logits = model.lm_head(
            hidden[
                :,
                start:end,
                :
            ]
        )

        targets = input_ids[
            :,
            start + 1:end + 1
        ]

        nll = F.cross_entropy(
            logits.float().reshape(
                -1,
                logits.shape[-1],
            ),
            targets.reshape(-1),
            reduction="none",
        )

        result[
            start + 1:end + 1
        ] = nll

        del (
            logits,
            targets,
            nll,
        )

    return result


# =============================================================================
# Capture exact raw LAMB importance while leaving model Vanilla
# =============================================================================

class LAMBImportanceCapture(
    DtRemappingBase
):
    """
    Reproduces AttentionBasedTopKMasking.obtain_vis_attn()
    but DOES NOT alter dt.

    Official formula:

      abs(
        einsum(
          B_history,
          C_recent,
          decay * dt_history
        )
      )

      subtract gamma * row-max
      clamp >= 0
      sum over head/query
    """

    def __init__(
        self,
        global_channel_indexes,
        per_channel_decays,
        layer_index,
        vis_window,
        gamma,
    ):
        super().__init__()

        self.global_channel_indexes = (
            global_channel_indexes
            .detach()
            .clone()
            .bool()
        )

        self.per_channel_decays = (
            per_channel_decays
            .detach()
            .clone()
        )

        self.layer_index = int(
            layer_index
        )

        self.vis_window = int(
            vis_window
        )

        self.gamma = float(
            gamma
        )

        self.raw_score = None

    def manipulate(
        self,
        variables,
    ):
        seq_len = int(
            variables[
                "dt"
            ].shape[1]
        )

        if (
            seq_len
            <=
            self.vis_window
        ):
            self.raw_score = torch.zeros(
                max(
                    0,
                    seq_len
                    -
                    self.vis_window
                ),
                dtype=torch.float32,
                device=variables[
                    "dt"
                ].device,
            )

            return (
                variables["dt"],
                variables["dt_bias"],
                variables["dt_softplus"],
                variables["A"],
                variables["B"],
                variables["C"],
            )

        dt = dt_bias_activate(
            variables
        )

        B = variables["B"]
        C = variables["C"]

        global_mask = (
            self.global_channel_indexes
        )

        attn = torch.einsum(
            "kc,qc,kh->hqk",
            B[
                0,
                :-self.vis_window,
                0,
            ],
            C[
                0,
                -self.vis_window:,
                0,
            ],
            (
                self.per_channel_decays[
                    global_mask
                ]
                *
                dt[
                    0,
                    :-self.vis_window,
                    global_mask
                ]
            ),
        ).abs()

        vis_attn = (
            attn
            -
            self.gamma
            *
            attn.max(
                dim=-1,
                keepdim=True,
            ).values
        ).clamp(
            min=0
        )

        self.raw_score = (
            vis_attn.sum(
                dim=[
                    0,
                    1,
                ]
            )
            .float()
            .detach()
        )

        # Return ORIGINAL variables => this pass remains Vanilla.
        return (
            variables["dt"],
            variables["dt_bias"],
            variables["dt_softplus"],
            variables["A"],
            variables["B"],
            variables["C"],
        )


# =============================================================================
# External per-layer mask
# =============================================================================

class ExternalPerLayerMask(
    DtRemappingBase
):
    def __init__(
        self,
        global_channel_indexes,
        layer_index,
    ):
        super().__init__()

        self.global_channel_indexes = (
            global_channel_indexes
            .detach()
            .clone()
            .bool()
        )

        self.layer_index = int(
            layer_index
        )

        self.mask = None

    def set_mask(
        self,
        mask,
    ):
        self.mask = (
            mask
            .detach()
            .clone()
            .bool()
        )

    def manipulate(
        self,
        variables,
    ):
        seq_len = int(
            variables[
                "dt"
            ].shape[1]
        )

        # Generated tokens update normally.
        if seq_len == 1:
            return (
                variables["dt"],
                variables["dt_bias"],
                variables["dt_softplus"],
                variables["A"],
                variables["B"],
                variables["C"],
            )

        if self.mask is None:
            raise RuntimeError(
                f"Layer {self.layer_index}: "
                "external mask unset"
            )

        if (
            self.mask.numel()
            !=
            seq_len
        ):
            raise RuntimeError(
                f"Layer {self.layer_index}: "
                f"mask={self.mask.numel()} "
                f"seq={seq_len}"
            )

        dt = dt_bias_activate(
            variables
        )

        selected = dt[
            :,
            :,
            self.global_channel_indexes,
        ]

        seq_mask = self.mask.to(
            dt.device
        )

        channel_mask = (
            seq_mask[
                :,
                None
            ]
            .expand(
                seq_len,
                selected.shape[-1],
            )
        )

        dt[
            :,
            :,
            self.global_channel_indexes,
        ] = torch.where(
            channel_mask[
                None,
                :,
                :
            ],
            selected,
            torch.zeros_like(
                selected
            ),
        )

        return (
            dt,
            None,
            False,
            variables["A"],
            variables["B"],
            variables["C"],
        )


# =============================================================================
# Routing controller
# =============================================================================

class RoutingController:
    def __init__(
        self,
        model,
        config,
        device,
    ):
        load_dt_remapping_from_config(
            model,
            config,
            device,
        )

        self.layers = list(
            model.backbone.layers
        )

        self.official = []
        self.global_masks = []
        self.decays = []
        self.gammas = []

        for layer in self.layers:
            mixer = layer.mixer

            cls = getattr(
                mixer,
                "dt_remap_cls",
                None,
            )

            if cls is None:
                raise RuntimeError(
                    "Official LAMB remapper missing"
                )

            if not hasattr(
                cls,
                "global_channel_indexes",
            ):
                raise RuntimeError(
                    "No global_channel_indexes "
                    "on official remapper"
                )

            if not hasattr(
                cls,
                "per_channel_decays",
            ):
                raise RuntimeError(
                    "No per_channel_decays "
                    "on official remapper"
                )

            self.official.append(
                {
                    "cls":
                        cls,

                    "func":
                        getattr(
                            mixer,
                            "dt_remap_func",
                            None,
                        ),

                    "cls2":
                        getattr(
                            mixer,
                            "dt_remap_cls2",
                            None,
                        ),

                    "func2":
                        getattr(
                            mixer,
                            "dt_remap_func2",
                            None,
                        ),
                }
            )

            self.global_masks.append(
                cls
                .global_channel_indexes
                .detach()
                .clone()
            )

            self.decays.append(
                cls
                .per_channel_decays
                .detach()
                .clone()
            )

            self.gammas.append(
                float(
                    getattr(
                        cls,
                        "gamma",
                        0.9,
                    )
                )
            )

        self.capture_remappers = [
            LAMBImportanceCapture(
                global_channel_indexes=(
                    self.global_masks[i]
                ),
                per_channel_decays=(
                    self.decays[i]
                ),
                layer_index=i,
                vis_window=int(
                    config[
                        "remap_kwargs"
                    ][
                        "vis_window"
                    ]
                ),
                gamma=(
                    self.gammas[i]
                ),
            )
            for i in range(
                len(
                    self.layers
                )
            )
        ]

        self.external_remappers = [
            ExternalPerLayerMask(
                global_channel_indexes=(
                    self.global_masks[i]
                ),
                layer_index=i,
            )
            for i in range(
                len(
                    self.layers
                )
            )
        ]

    def official_lamb(self):
        for layer, saved in zip(
            self.layers,
            self.official,
        ):
            mixer = layer.mixer

            mixer.dt_remap_cls = (
                saved["cls"]
            )

            mixer.dt_remap_func = (
                saved["func"]
            )

            mixer.dt_remap_cls2 = (
                saved["cls2"]
            )

            mixer.dt_remap_func2 = (
                saved["func2"]
            )

    def vanilla(self):
        for layer in self.layers:
            mixer = layer.mixer

            mixer.dt_remap_func = None
            mixer.dt_remap_func2 = None

    def capture_lamb_scores(self):
        for layer, remapper in zip(
            self.layers,
            self.capture_remappers,
        ):
            remapper.raw_score = None

            mixer = layer.mixer

            mixer.dt_remap_cls = (
                remapper
            )

            mixer.dt_remap_func = (
                remapper.manipulate
            )

            mixer.dt_remap_cls2 = None
            mixer.dt_remap_func2 = None

    def external_masks(
        self,
        masks,
    ):
        if (
            len(masks)
            !=
            len(
                self.layers
            )
        ):
            raise RuntimeError(
                "Number of masks != layers"
            )

        for (
            layer,
            remapper,
            mask,
        ) in zip(
            self.layers,
            self.external_remappers,
            masks,
        ):
            remapper.set_mask(
                mask
            )

            mixer = layer.mixer

            mixer.dt_remap_cls = (
                remapper
            )

            mixer.dt_remap_func = (
                remapper.manipulate
            )

            mixer.dt_remap_cls2 = None
            mixer.dt_remap_func2 = None


# =============================================================================
# Intermediate layer state capture for bounded-context Surprisal
# =============================================================================

def layer_state_from_output(
    output,
):
    if isinstance(
        output,
        tuple,
    ):
        hidden = output[0]

        residual = (
            output[1]
            if len(output) > 1
            else None
        )

        if (
            residual is not None
            and
            torch.is_tensor(
                residual
            )
        ):
            return (
                hidden.float()
                +
                residual.float()
            )

        return hidden.float()

    return output.float()


def forward_capture_anchor_states(
    model,
    layers,
    input_ids,
    anchor_layers,
):
    """
    anchor_layers are 1-indexed.
    Returns dict:
        anchor_layer -> [seq, hidden]
    """

    buffer = {}
    handles = []

    wanted = {
        int(x) - 1
        for x in anchor_layers
    }

    for zero_idx in wanted:
        layer = layers[
            zero_idx
        ]

        def make_hook(
            idx
        ):
            def hook(
                module,
                inputs,
                output,
            ):
                state = (
                    layer_state_from_output(
                        output
                    )
                )

                buffer[
                    idx + 1
                ] = (
                    state[
                        0
                    ]
                    .detach()
                )

            return hook

        handles.append(
            layer.register_forward_hook(
                make_hook(
                    zero_idx
                )
            )
        )

    try:
        with torch.inference_mode():
            _ = model.backbone(
                input_ids
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        x
        for x in anchor_layers
        if x not in buffer
    ]

    if missing:
        raise RuntimeError(
            f"Missing anchor layers: "
            f"{missing}"
        )

    return buffer


def nll_from_layer_states(
    model,
    states,
    predecessor_positions,
    target_ids,
    chunk_size,
):
    """
    Apply final norm + lm_head as a logit lens.

    states:
        [window_seq, hidden]

    predecessor_positions:
        positions whose hidden states predict target_ids.
    """

    result = []

    for start in range(
        0,
        len(
            predecessor_positions
        ),
        int(chunk_size),
    ):
        end = min(
            start
            +
            int(chunk_size),
            len(
                predecessor_positions
            ),
        )

        idx = torch.tensor(
            predecessor_positions[
                start:end
            ],
            dtype=torch.long,
            device=states.device,
        )

        x = states.index_select(
            0,
            idx,
        )

        # Restore model dtype for the native RMSNorm/head.
        x = x.to(
            dtype=next(
                model.parameters()
            ).dtype
        )

        normalized = (
            model.backbone.norm_f(
                x
            )
        )

        logits = model.lm_head(
            normalized
        )

        target = target_ids[
            start:end
        ]

        loss = F.cross_entropy(
            logits.float(),
            target,
            reduction="none",
        )

        result.append(
            loss.detach()
        )

        del (
            idx,
            x,
            normalized,
            logits,
            target,
            loss,
        )

    return torch.cat(
        result,
        dim=0,
    )


# =============================================================================
# Bounded-context layer-band Surprisal
# =============================================================================

def compute_bounded_layer_surprisal(
    model,
    controller,
    input_ids,
    anchor_layers,
    core_size,
    left_context,
    logit_chunk_size,
):
    """
    Each core block is evaluated after a fresh recurrent reset.

    For core:
        [c, c+core_size)

    forward window:
        [max(0,c-left_context), c+core_size)

    Therefore Surprisal cannot depend on arbitrarily old state.

    With defaults:
        core_size=1024
        left_context=128

    max causal span is about 1152 tokens.
    """

    controller.vanilla()

    seq_len = int(
        input_ids.shape[1]
    )

    device = input_ids.device

    scores = torch.zeros(
        (
            len(
                anchor_layers
            ),
            seq_len,
        ),
        dtype=torch.float32,
        device=device,
    )

    counts = torch.zeros(
        seq_len,
        dtype=torch.int32,
        device=device,
    )

    anchor_to_idx = {
        int(layer):
            i
        for i, layer
        in enumerate(
            anchor_layers
        )
    }

    for core_start in range(
        0,
        seq_len,
        int(core_size),
    ):
        core_end = min(
            core_start
            +
            int(core_size),
            seq_len,
        )

        window_start = max(
            0,
            core_start
            -
            int(left_context),
        )

        window_end = (
            core_end
        )

        window_ids = input_ids[
            :,
            window_start:window_end,
        ]

        captured = (
            forward_capture_anchor_states(
                model=model,
                layers=controller.layers,
                input_ids=window_ids,
                anchor_layers=anchor_layers,
            )
        )

        target_global_positions = list(
            range(
                max(
                    1,
                    core_start,
                ),
                core_end,
            )
        )

        if not target_global_positions:
            del (
                window_ids,
                captured,
            )
            continue

        predecessor_positions = [
            (
                global_pos
                -
                window_start
                -
                1
            )
            for global_pos
            in target_global_positions
        ]

        target_ids = input_ids[
            0,
            target_global_positions,
        ]

        for layer in anchor_layers:
            anchor_idx = (
                anchor_to_idx[
                    int(layer)
                ]
            )

            loss = nll_from_layer_states(
                model=model,
                states=captured[
                    int(layer)
                ],
                predecessor_positions=(
                    predecessor_positions
                ),
                target_ids=target_ids,
                chunk_size=(
                    logit_chunk_size
                ),
            )

            scores[
                anchor_idx,
                target_global_positions,
            ] = loss

        counts[
            target_global_positions
        ] += 1

        del (
            window_ids,
            captured,
            target_ids,
        )

        torch.cuda.empty_cache()

    if seq_len > 1:
        if not torch.all(
            counts[
                1:
            ]
            ==
            1
        ):
            raise RuntimeError(
                "Bounded Surprisal coverage "
                "invariant failed"
            )

        # Position 0 has no causal predecessor.
        scores[
            :,
            0
        ] = scores[
            :,
            1
        ]

    return scores


# =============================================================================
# Layer-band mapping
# =============================================================================

def layer_to_anchor(
    layer_1indexed,
    anchors,
):
    for anchor in anchors:
        if (
            layer_1indexed
            <=
            anchor
        ):
            return anchor

    return anchors[-1]


# =============================================================================
# Fusion calibration
# =============================================================================

def dense_rms_normalize(
    x,
    eps=1e-6,
):
    """
    Preserve sign and magnitude ordering.
    No quantile, no clipping.
    """

    rms = torch.sqrt(
        torch.mean(
            x.float()
            *
            x.float()
        )
    )

    return (
        x.float()
        /
        (
            rms
            +
            eps
        )
    )


def sparse_positive_rms_normalize(
    x,
    eps=1e-6,
):
    """
    Critical property:
        x == 0  -->  output == 0

    Positive LAMB geometry is preserved exactly.
    """

    x = x.float()

    positive = (
        x > 0
    )

    output = torch.zeros_like(
        x
    )

    if not positive.any():
        return output

    rms = torch.sqrt(
        torch.mean(
            x[
                positive
            ]
            *
            x[
                positive
            ]
        )
    )

    output[
        positive
    ] = (
        x[
            positive
        ]
        /
        (
            rms
            +
            eps
        )
    )

    return output


# =============================================================================
# Build all selector masks
# =============================================================================

def build_all_masks(
    seq_len,
    full_surprisal,
    bounded_surprisal,
    lamb_raw_scores,
    anchor_layers,
    n_layers,
    norm_window,
    topk,
    vis_window,
    pooling_kernel,
    lambda_surprisal,
):
    device = (
        full_surprisal.device
    )

    # -------------------------------------------------------------------------
    # 1. Canonical full-context shared Surprisal
    # -------------------------------------------------------------------------

    full_z = rolling_mean_std_zscore(
        full_surprisal,
        norm_window,
    )

    (
        full_shared_mask,
        full_shared_pooled,
    ) = build_mask_from_position_scores(
        position_score=full_z,
        seq_len=seq_len,
        topk=topk,
        vis_window=vis_window,
        pooling_kernel=pooling_kernel,
    )

    canonical_full_masks = [
        full_shared_mask
        for _ in range(
            n_layers
        )
    ]

    # -------------------------------------------------------------------------
    # 2. Bounded Surprisal normalization per anchor
    # -------------------------------------------------------------------------

    bounded_z = []

    bounded_pooled = []

    bounded_masks_by_anchor = {}

    for anchor_idx, anchor in enumerate(
        anchor_layers
    ):
        z = rolling_mean_std_zscore(
            bounded_surprisal[
                anchor_idx
            ],
            norm_window,
        )

        (
            mask,
            pooled,
        ) = build_mask_from_position_scores(
            position_score=z,
            seq_len=seq_len,
            topk=topk,
            vis_window=vis_window,
            pooling_kernel=pooling_kernel,
        )

        bounded_z.append(
            z
        )

        bounded_pooled.append(
            pooled
        )

        bounded_masks_by_anchor[
            int(anchor)
        ] = mask

    bounded_z = torch.stack(
        bounded_z,
        dim=0,
    )

    # -------------------------------------------------------------------------
    # 3. Bounded shared = final anchor only
    # -------------------------------------------------------------------------

    final_anchor = int(
        anchor_layers[-1]
    )

    bounded_shared_mask = (
        bounded_masks_by_anchor[
            final_anchor
        ]
    )

    bounded_shared_masks = [
        bounded_shared_mask
        for _ in range(
            n_layers
        )
    ]

    # -------------------------------------------------------------------------
    # 4. Bounded layer-band
    # -------------------------------------------------------------------------

    bounded_band_masks = []

    for layer_idx in range(
        n_layers
    ):
        layer_number = (
            layer_idx + 1
        )

        anchor = layer_to_anchor(
            layer_number,
            anchor_layers,
        )

        bounded_band_masks.append(
            bounded_masks_by_anchor[
                anchor
            ]
        )

    # -------------------------------------------------------------------------
    # 5. Exact pooled LAMB scores + sparse-preserving fusion
    # -------------------------------------------------------------------------

    fusion_masks = []

    lamb_pooled_scores = []

    fusion_abs_s_share = []

    lamb_zero_fractions = []

    for layer_idx in range(
        n_layers
    ):
        if (
            seq_len
            <
            int(topk)
            +
            int(vis_window)
        ):
            fusion_masks.append(
                torch.ones(
                    seq_len,
                    dtype=torch.bool,
                    device=device,
                )
            )

            lamb_pooled_scores.append(
                None
            )

            fusion_abs_s_share.append(
                float("nan")
            )

            lamb_zero_fractions.append(
                float("nan")
            )

            continue

        lamb_raw = (
            lamb_raw_scores[
                layer_idx
            ]
        )

        if lamb_raw is None:
            raise RuntimeError(
                f"No LAMB raw score "
                f"for layer {layer_idx+1}"
            )

        lamb_pool = official_mean_pool(
            lamb_raw,
            pooling_kernel,
        )

        lamb_pooled_scores.append(
            lamb_pool
        )

        lamb_zero_fractions.append(
            float(
                (
                    lamb_pool
                    ==
                    0
                )
                .float()
                .mean()
                .item()
            )
        )

        layer_number = (
            layer_idx + 1
        )

        anchor = layer_to_anchor(
            layer_number,
            anchor_layers,
        )

        anchor_idx = (
            anchor_layers.index(
                anchor
            )
        )

        s_pool = (
            bounded_pooled[
                anchor_idx
            ]
        )

        if (
            s_pool is None
            or
            s_pool.shape
            !=
            lamb_pool.shape
        ):
            raise RuntimeError(
                f"Fusion pooled-score "
                f"shape mismatch at "
                f"layer {layer_number}: "
                f"S={None if s_pool is None else tuple(s_pool.shape)} "
                f"L={tuple(lamb_pool.shape)}"
            )

        s_norm = dense_rms_normalize(
            s_pool
        )

        l_norm = (
            sparse_positive_rms_normalize(
                lamb_pool
            )
        )

        fused = (
            float(
                lambda_surprisal
            )
            *
            s_norm
            +
            (
                1.0
                -
                float(
                    lambda_surprisal
                )
            )
            *
            l_norm
        )

        mask = (
            build_official_budget_mask_from_pooled(
                pooled_score=fused,
                seq_len=seq_len,
                topk=topk,
                vis_window=vis_window,
                device=device,
            )
        )

        fusion_masks.append(
            mask
        )

        selected = mask[
            :fused.numel()
        ]

        if selected.any():
            s_component = (
                float(
                    lambda_surprisal
                )
                *
                s_norm[
                    selected
                ]
            ).abs()

            l_component = (
                (
                    1.0
                    -
                    float(
                        lambda_surprisal
                    )
                )
                *
                l_norm[
                    selected
                ]
            ).abs()

            denom = (
                s_component.sum()
                +
                l_component.sum()
            )

            if denom.item() > 0:
                share = float(
                    (
                        s_component.sum()
                        /
                        denom
                    ).item()
                )
            else:
                share = float(
                    "nan"
                )
        else:
            share = float(
                "nan"
            )

        fusion_abs_s_share.append(
            share
        )

    return {
        "canonical_full_masks":
            canonical_full_masks,

        "bounded_shared_masks":
            bounded_shared_masks,

        "bounded_band_masks":
            bounded_band_masks,

        "fusion_masks":
            fusion_masks,

        "full_z":
            full_z,

        "bounded_z":
            bounded_z,

        "fusion_abs_surprisal_share":
            fusion_abs_s_share,

        "lamb_pooled_zero_fraction":
            lamb_zero_fractions,
    }


# =============================================================================
# Generation
# =============================================================================

def generate_answer(
    model,
    tokenizer,
    input_ids,
    max_gen,
):
    prompt_len = int(
        input_ids.shape[1]
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=(
                prompt_len
                +
                int(max_gen)
            ),
            top_k=1,
            temperature=1.0,
            output_scores=False,
            enable_timing=False,
            eos_token_id=torch.tensor(
                tokenizer.eos_token_id,
                device=input_ids.device,
            ),
        )

    prediction = tokenizer.decode(
        output_ids[
            0,
            prompt_len:
        ],
        skip_special_tokens=True,
    )

    del output_ids

    return prediction


# =============================================================================
# Plotting
# =============================================================================

def plot_results(
    dataset_summary,
    datasets,
    output_dir,
):
    plot_dir = (
        Path(
            output_dir
        )
        /
        "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Dataset score plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            13,
            6,
        )
    )

    x = np.arange(
        len(
            datasets
        )
    )

    width = (
        0.82
        /
        len(
            ARMS
        )
    )

    for arm_idx, arm in enumerate(
        ARMS
    ):
        values = [
            dataset_summary[
                d
            ][
                arm
            ][
                "mean_score"
            ]
            for d in datasets
        ]

        offset = (
            arm_idx
            -
            (
                len(
                    ARMS
                )
                -
                1
            )
            /
            2
        ) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            label=arm,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        datasets,
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "LongBench-E score"
    )

    ax.set_title(
        "Canonical Surprisal Final Suite"
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        plot_dir
        /
        "dataset_scores.png",
        dpi=240,
        bbox_inches="tight",
    )

    fig.savefig(
        plot_dir
        /
        "dataset_scores.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------
    # Delta to LAMB
    # -------------------------------------------------------------------------

    candidate_arms = [
        x
        for x in ARMS
        if x
        !=
        "original_lamb"
    ]

    fig, ax = plt.subplots(
        figsize=(
            13,
            6,
        )
    )

    width = (
        0.82
        /
        len(
            candidate_arms
        )
    )

    for arm_idx, arm in enumerate(
        candidate_arms
    ):
        values = [
            dataset_summary[
                d
            ][
                arm
            ][
                "delta_lamb"
            ]
            for d in datasets
        ]

        offset = (
            arm_idx
            -
            (
                len(
                    candidate_arms
                )
                -
                1
            )
            /
            2
        ) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            label=arm,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        datasets,
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Score difference vs Original LAMB"
    )

    ax.set_title(
        "Surprisal Variants vs Original LAMB"
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        plot_dir
        /
        "delta_vs_lamb.png",
        dpi=240,
        bbox_inches="tight",
    )

    fig.savefig(
        plot_dir
        /
        "delta_vs_lamb.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Aggregate
# =============================================================================

def aggregate(
    output_dir,
    datasets,
):
    output_dir = Path(
        output_dir
    )

    all_rows = []

    for dataset in datasets:
        path = (
            output_dir
            /
            "generation"
            /
            f"{dataset}.jsonl"
        )

        rows = read_jsonl(
            path
        )

        all_rows.extend(
            rows
        )

    by_dataset = defaultdict(
        list
    )

    for row in all_rows:
        by_dataset[
            row[
                "dataset"
            ]
        ].append(
            row
        )

    summary = {}

    for dataset in datasets:
        rows = by_dataset[
            dataset
        ]

        if not rows:
            continue

        summary[
            dataset
        ] = {}

        lamb_scores = np.asarray(
            [
                r[
                    "arms"
                ][
                    "original_lamb"
                ][
                    "score"
                ]
                for r in rows
            ],
            dtype=np.float64,
        )

        for arm in ARMS:
            scores = np.asarray(
                [
                    r[
                        "arms"
                    ][
                        arm
                    ][
                        "score"
                    ]
                    for r in rows
                ],
                dtype=np.float64,
            )

            delta = (
                scores
                -
                lamb_scores
            )

            summary[
                dataset
            ][
                arm
            ] = {
                "n":
                    len(
                        rows
                    ),

                "mean_score":
                    float(
                        scores.mean()
                    ),

                "delta_lamb":
                    float(
                        delta.mean()
                    ),

                "wins":
                    int(
                        np.sum(
                            delta > 0
                        )
                    ),

                "ties":
                    int(
                        np.sum(
                            delta == 0
                        )
                    ),

                "losses":
                    int(
                        np.sum(
                            delta < 0
                        )
                    ),
            }

    macro = {}

    valid_datasets = [
        d
        for d in datasets
        if d in summary
    ]

    for arm in ARMS:
        macro[
            arm
        ] = {
            "macro_score":
                finite_mean(
                    [
                        summary[
                            d
                        ][
                            arm
                        ][
                            "mean_score"
                        ]
                        for d
                        in valid_datasets
                    ]
                ),

            "macro_delta_lamb":
                finite_mean(
                    [
                        summary[
                            d
                        ][
                            arm
                        ][
                            "delta_lamb"
                        ]
                        for d
                        in valid_datasets
                    ]
                ),
        }

    write_json(
        output_dir
        /
        "dataset_summary.json",
        {
            "datasets":
                summary,

            "macro":
                macro,
        },
    )

    # CSV
    csv_path = (
        output_dir
        /
        "dataset_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "dataset",
                "arm",
                "n",
                "mean_score",
                "delta_lamb",
                "wins",
                "ties",
                "losses",
            ]
        )

        for dataset in valid_datasets:
            for arm in ARMS:
                x = summary[
                    dataset
                ][
                    arm
                ]

                writer.writerow(
                    [
                        dataset,
                        arm,
                        x["n"],
                        x[
                            "mean_score"
                        ],
                        x[
                            "delta_lamb"
                        ],
                        x["wins"],
                        x["ties"],
                        x["losses"],
                    ]
                )

    plot_results(
        dataset_summary=summary,
        datasets=valid_datasets,
        output_dir=output_dir,
    )

    print()
    print(
        "=" * 142
    )
    print(
        "CANONICAL SURPRISAL FINAL SUITE — SUMMARY"
    )
    print(
        "=" * 142
    )

    print(
        f"{'Dataset':24s}"
        f"{'LAMB':>10s}"
        f"{'Canon-S':>10s}"
        f"{'Bound-S':>10s}"
        f"{'Band-S':>10s}"
        f"{'Fusion':>10s}"
        f"{'Fusion Δ':>11s}"
        f"{'Fusion W/T/L':>15s}"
    )

    print(
        "-" * 142
    )

    for dataset in valid_datasets:
        s = summary[
            dataset
        ]

        fusion = s[
            "fusion_all"
        ]

        print(
            f"{dataset:24s}"
            f"{s['original_lamb']['mean_score']:10.3f}"
            f"{s['canonical_full_shared']['mean_score']:10.3f}"
            f"{s['bounded_local_shared']['mean_score']:10.3f}"
            f"{s['bounded_local_band']['mean_score']:10.3f}"
            f"{fusion['mean_score']:10.3f}"
            f"{fusion['delta_lamb']:+11.3f}"
            f"{fusion['wins']}/{fusion['ties']}/{fusion['losses']:>5d}"
        )

    print(
        "-" * 142
    )

    print(
        f"{'MACRO':24s}"
        f"{macro['original_lamb']['macro_score']:10.3f}"
        f"{macro['canonical_full_shared']['macro_score']:10.3f}"
        f"{macro['bounded_local_shared']['macro_score']:10.3f}"
        f"{macro['bounded_local_band']['macro_score']:10.3f}"
        f"{macro['fusion_all']['macro_score']:10.3f}"
        f"{macro['fusion_all']['macro_delta_lamb']:+11.3f}"
    )

    print(
        "=" * 142
    )

    print()
    print(
        "Output:",
        output_dir,
    )

    print(
        "Plots:",
        output_dir
        /
        "plots",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--datasets",
        default=",".join(
            DEFAULT_DATASETS
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--norm-window",
        type=int,
        default=65,
    )

    parser.add_argument(
        "--anchor-layers",
        default="8,16,24,32,40,48",
    )

    parser.add_argument(
        "--bounded-core-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--bounded-left-context",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--logit-chunk-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lambda-surprisal",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/Workspace/LMSelfBenchResult/"
            "experiments/"
            "surprisal_teacher_audit_20260902/"
            "results/"
            "canonical_surprisal_final_suite_20260902"
        ),
    )

    parser.add_argument(
        "--analysis-only",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_all(
        args.seed
    )

    disable_progress_bar()

    datasets = [
        x.strip()
        for x in args.datasets.split(",")
        if x.strip()
    ]

    anchor_layers = [
        int(x)
        for x in args.anchor_layers.split(",")
        if x.strip()
    ]

    if (
        not anchor_layers
        or
        anchor_layers[-1]
        !=
        48
    ):
        raise ValueError(
            "Anchor layers must end at 48 "
            "for mamba2-1.3b"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            Path(
                LAMB_ROOT
            )
            /
            config_path
        )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    remap_kwargs = config[
        "remap_kwargs"
    ]

    if (
        remap_kwargs[
            "pooling"
        ]
        !=
        "mean"
    ):
        raise RuntimeError(
            "This suite expects official "
            "LAMB mean pooling"
        )

    topk = int(
        remap_kwargs[
            "attn_topk"
        ]
    )

    vis_window = int(
        remap_kwargs[
            "vis_window"
        ]
    )

    pooling_kernel = int(
        remap_kwargs[
            "pooling_kernel"
        ]
    )

    manifest = {
        "experiment":
            "Canonical Surprisal Final Suite",

        "model":
            MODEL_NAME,

        "datasets":
            datasets,

        "samples_per_dataset":
            args.limit,

        "arms":
            ARMS,

        "official_lamb_budget":
            {
                "topk":
                    topk,

                "vis_window":
                    vis_window,

                "pooling":
                    "mean",

                "pooling_kernel":
                    pooling_kernel,
            },

        "canonical_surprisal":
            (
                "causal NLL -> rolling mean/std zscore -> "
                "official LAMB mean pooling -> TopK"
            ),

        "bounded_context":
            {
                "core_size":
                    args.bounded_core_size,

                "left_context":
                    args.bounded_left_context,

                "maximum_effective_span":
                    (
                        args.bounded_core_size
                        +
                        args.bounded_left_context
                    ),
            },

        "anchor_layers":
            anchor_layers,

        "layer_band_policy":
            (
                "layer <= next anchor "
                "uses that anchor Surprisal"
            ),

        "fusion":
            {
                "lambda_surprisal":
                    args.lambda_surprisal,

                "surprisal_calibration":
                    "dense RMS after official pooling",

                "lamb_calibration":
                    (
                        "positive-only RMS after official pooling; "
                        "exact zeros remain zero"
                    ),

                "quantile":
                    False,

                "clipping":
                    False,
            },
    }

    write_json(
        output_dir
        /
        "experiment_manifest.json",
        manifest,
    )

    if args.analysis_only:
        aggregate(
            output_dir,
            datasets,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = (
        f"cuda:{args.device}"
    )

    print(
        "Loading tokenizer...",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_NAME,
            clean_up_tokenization_spaces=False,
        )
    )

    print(
        "Loading Mamba2-1.3B...",
        flush=True,
    )

    model = (
        MambaLMHeadModel
        .from_pretrained(
            MODEL_NAME,
            device=device,
            dtype=torch.bfloat16,
        )
    )

    model.eval()

    controller = RoutingController(
        model,
        config,
        device,
    )

    if (
        len(
            controller.layers
        )
        !=
        48
    ):
        raise RuntimeError(
            f"Expected 48 Mamba layers, got "
            f"{len(controller.layers)}"
        )

    generation_dir = (
        output_dir
        /
        "generation"
    )

    diagnostics_path = (
        output_dir
        /
        "selector_diagnostics.csv"
    )

    diagnostics_exists = (
        diagnostics_path.exists()
    )

    diagnostics_file = open(
        diagnostics_path,
        "a",
        newline="",
        encoding="utf-8",
    )

    diagnostics_writer = csv.writer(
        diagnostics_file
    )

    if not diagnostics_exists:
        diagnostics_writer.writerow(
            [
                "dataset",
                "sample_index",
                "prompt_tokens",
                "mean_jaccard_fullS_vs_LAMB",
                "mean_jaccard_boundShared_vs_LAMB",
                "mean_jaccard_bandS_vs_LAMB",
                "mean_jaccard_fusion_vs_LAMB",
                "mean_jaccard_fusion_vs_bandS",
                "mean_lamb_pooled_zero_fraction",
                "mean_fusion_abs_surprisal_share",
            ]
        )

        diagnostics_file.flush()

    try:
        for dataset in datasets:
            print()
            print(
                "=" * 108
            )

            print(
                "DATASET:",
                dataset,
            )

            print(
                "=" * 108
            )

            ds = load_dataset(
                "THUDM/LongBench",
                f"{dataset}_e",
                split="test",
                trust_remote_code=True,
            )

            count = min(
                args.limit,
                len(ds),
            )

            out_path = (
                generation_dir
                /
                f"{dataset}.jsonl"
            )

            existing = read_jsonl(
                out_path
            )

            done_indices = {
                int(
                    x[
                        "sample_index"
                    ]
                )
                for x in existing
            }

            bar = tqdm(
                range(count),
                desc=dataset,
                dynamic_ncols=True,
            )

            for idx in bar:
                if idx in done_indices:
                    continue

                sample = ds[
                    idx
                ]

                prompt = (
                    dataset2prompt[
                        dataset
                    ].format(
                        **sample
                    )
                )

                if prompt.endswith(" "):
                    prompt = prompt[:-1]

                input_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=False,
                )[
                    "input_ids"
                ].to(
                    device
                )

                seq_len = int(
                    input_ids.shape[1]
                )

                # =============================================================
                # FIRST PASS:
                # Vanilla forward + exact raw LAMB importance capture
                # + full-context canonical Surprisal
                # =============================================================

                controller.capture_lamb_scores()

                with torch.inference_mode():
                    hidden = model.backbone(
                        input_ids
                    )

                    full_surprisal = (
                        compute_final_surprisal(
                            model=model,
                            hidden=hidden,
                            input_ids=input_ids,
                            chunk_size=(
                                args.logit_chunk_size
                            ),
                        )
                    )

                del hidden

                lamb_raw_scores = []

                for layer_idx, capture in enumerate(
                    controller.capture_remappers
                ):
                    if (
                        capture.raw_score
                        is None
                    ):
                        raise RuntimeError(
                            f"{dataset}/{idx}: "
                            f"LAMB capture missing "
                            f"layer {layer_idx+1}"
                        )

                    lamb_raw_scores.append(
                        capture.raw_score
                    )

                # =============================================================
                # BOUNDED-CONTEXT LAYER-BAND SURPRISAL
                # =============================================================

                bounded_surprisal = (
                    compute_bounded_layer_surprisal(
                        model=model,
                        controller=controller,
                        input_ids=input_ids,
                        anchor_layers=(
                            anchor_layers
                        ),
                        core_size=(
                            args.bounded_core_size
                        ),
                        left_context=(
                            args.bounded_left_context
                        ),
                        logit_chunk_size=(
                            args.logit_chunk_size
                        ),
                    )
                )

                # =============================================================
                # BUILD ALL FOUR NEW SELECTORS
                # =============================================================

                selector = build_all_masks(
                    seq_len=seq_len,
                    full_surprisal=(
                        full_surprisal
                    ),
                    bounded_surprisal=(
                        bounded_surprisal
                    ),
                    lamb_raw_scores=(
                        lamb_raw_scores
                    ),
                    anchor_layers=(
                        anchor_layers
                    ),
                    n_layers=len(
                        controller.layers
                    ),
                    norm_window=(
                        args.norm_window
                    ),
                    topk=topk,
                    vis_window=vis_window,
                    pooling_kernel=(
                        pooling_kernel
                    ),
                    lambda_surprisal=(
                        args.lambda_surprisal
                    ),
                )

                # =============================================================
                # Reconstruct official LAMB token masks for DIAGNOSTICS ONLY
                # =============================================================

                reconstructed_lamb_masks = []

                for layer_idx in range(
                    len(
                        controller.layers
                    )
                ):
                    if (
                        seq_len
                        <
                        topk
                        +
                        vis_window
                    ):
                        lm = torch.ones(
                            seq_len,
                            dtype=torch.bool,
                            device=device,
                        )

                    else:
                        lp = official_mean_pool(
                            lamb_raw_scores[
                                layer_idx
                            ],
                            pooling_kernel,
                        )

                        lm = (
                            build_official_budget_mask_from_pooled(
                                pooled_score=lp,
                                seq_len=seq_len,
                                topk=topk,
                                vis_window=vis_window,
                                device=device,
                            )
                        )

                    reconstructed_lamb_masks.append(
                        lm
                    )

                # =============================================================
                # Generation: all five arms
                # =============================================================

                arm_masks = {
                    "canonical_full_shared":
                        selector[
                            "canonical_full_masks"
                        ],

                    "bounded_local_shared":
                        selector[
                            "bounded_shared_masks"
                        ],

                    "bounded_local_band":
                        selector[
                            "bounded_band_masks"
                        ],

                    "fusion_all":
                        selector[
                            "fusion_masks"
                        ],
                }

                arm_results = {}

                # Original official LAMB
                controller.official_lamb()

                pred = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    max_gen=int(
                        dataset2maxgenlen[
                            dataset
                        ]
                    ),
                )

                arm_results[
                    "original_lamb"
                ] = {
                    "prediction":
                        pred,

                    "score":
                        score_prediction(
                            dataset=dataset,
                            prediction=pred,
                            answers=sample[
                                "answers"
                            ],
                            all_classes=sample[
                                "all_classes"
                            ],
                        ),
                }

                # Four candidate arms
                for arm in [
                    "canonical_full_shared",
                    "bounded_local_shared",
                    "bounded_local_band",
                    "fusion_all",
                ]:
                    controller.external_masks(
                        arm_masks[
                            arm
                        ]
                    )

                    pred = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        max_gen=int(
                            dataset2maxgenlen[
                                dataset
                            ]
                        ),
                    )

                    arm_results[
                        arm
                    ] = {
                        "prediction":
                            pred,

                        "score":
                            score_prediction(
                                dataset=dataset,
                                prediction=pred,
                                answers=sample[
                                    "answers"
                                ],
                                all_classes=sample[
                                    "all_classes"
                                ],
                            ),
                    }

                # =============================================================
                # Diagnostics
                # =============================================================

                full_j = []
                shared_j = []
                band_j = []
                fusion_lamb_j = []
                fusion_band_j = []

                for layer_idx in range(
                    len(
                        controller.layers
                    )
                ):
                    lm = (
                        reconstructed_lamb_masks[
                            layer_idx
                        ]
                    )

                    full_j.append(
                        jaccard(
                            selector[
                                "canonical_full_masks"
                            ][
                                layer_idx
                            ],
                            lm,
                        )
                    )

                    shared_j.append(
                        jaccard(
                            selector[
                                "bounded_shared_masks"
                            ][
                                layer_idx
                            ],
                            lm,
                        )
                    )

                    band_j.append(
                        jaccard(
                            selector[
                                "bounded_band_masks"
                            ][
                                layer_idx
                            ],
                            lm,
                        )
                    )

                    fusion_lamb_j.append(
                        jaccard(
                            selector[
                                "fusion_masks"
                            ][
                                layer_idx
                            ],
                            lm,
                        )
                    )

                    fusion_band_j.append(
                        jaccard(
                            selector[
                                "fusion_masks"
                            ][
                                layer_idx
                            ],
                            selector[
                                "bounded_band_masks"
                            ][
                                layer_idx
                            ],
                        )
                    )

                diagnostics = {
                    "mean_jaccard_fullS_vs_LAMB":
                        finite_mean(
                            full_j
                        ),

                    "mean_jaccard_boundShared_vs_LAMB":
                        finite_mean(
                            shared_j
                        ),

                    "mean_jaccard_bandS_vs_LAMB":
                        finite_mean(
                            band_j
                        ),

                    "mean_jaccard_fusion_vs_LAMB":
                        finite_mean(
                            fusion_lamb_j
                        ),

                    "mean_jaccard_fusion_vs_bandS":
                        finite_mean(
                            fusion_band_j
                        ),

                    "mean_lamb_pooled_zero_fraction":
                        finite_mean(
                            selector[
                                "lamb_pooled_zero_fraction"
                            ]
                        ),

                    "mean_fusion_abs_surprisal_share":
                        finite_mean(
                            selector[
                                "fusion_abs_surprisal_share"
                            ]
                        ),
                }

                diagnostics_writer.writerow(
                    [
                        dataset,
                        idx,
                        seq_len,
                        diagnostics[
                            "mean_jaccard_fullS_vs_LAMB"
                        ],
                        diagnostics[
                            "mean_jaccard_boundShared_vs_LAMB"
                        ],
                        diagnostics[
                            "mean_jaccard_bandS_vs_LAMB"
                        ],
                        diagnostics[
                            "mean_jaccard_fusion_vs_LAMB"
                        ],
                        diagnostics[
                            "mean_jaccard_fusion_vs_bandS"
                        ],
                        diagnostics[
                            "mean_lamb_pooled_zero_fraction"
                        ],
                        diagnostics[
                            "mean_fusion_abs_surprisal_share"
                        ],
                    ]
                )

                diagnostics_file.flush()

                row = {
                    "dataset":
                        dataset,

                    "sample_index":
                        idx,

                    "_id":
                        sample.get(
                            "_id"
                        ),

                    "prompt_tokens":
                        seq_len,

                    "answers":
                        sample[
                            "answers"
                        ],

                    "arms":
                        arm_results,

                    "selector_diagnostics":
                        diagnostics,
                }

                append_jsonl(
                    out_path,
                    row,
                )

                bar.set_postfix(
                    L=(
                        f"{arm_results['original_lamb']['score']:.1f}"
                    ),
                    F=(
                        f"{arm_results['fusion_all']['score']:.1f}"
                    ),
                )

                del (
                    input_ids,
                    full_surprisal,
                    bounded_surprisal,
                    lamb_raw_scores,
                    selector,
                    reconstructed_lamb_masks,
                )

                torch.cuda.empty_cache()

            del ds

    finally:
        diagnostics_file.close()

    # =========================================================================
    # Automatic aggregation + plots
    # =========================================================================

    aggregate(
        output_dir,
        datasets,
    )


if __name__ == "__main__":
    main()
