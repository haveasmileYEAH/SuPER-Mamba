#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import disable_progress_bar, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


# =====================================================================
# Existing audited implementation
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAMB_ROOT = Path(os.environ.get(
    "LAMB_ROOT", str(PROJECT_ROOT / "third_party/lamb")
))
ARCHIVE_SCRIPTS = PROJECT_ROOT / "super_mamba"

sys.path.insert(0, str(LAMB_ROOT))
sys.path.insert(0, str(ARCHIVE_SCRIPTS))

import surprisal_pipeline as six

base = six.base

from tasks.longbench import scorer_e


# =====================================================================
# Fixed experiment settings
# =====================================================================

MODEL_NAME = "state-spaces/mamba2-1.3b"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"

DATASET = "trec"

VIS_WINDOW = 32

TOPKS = [
    256,
    512,
    768,
    1024,
    1536,   # x
    2048,
    3072,   # y
]

POOLING_KERNELS = [
    3,
    5,
    7,
    9,
]

POOLING_METHODS = [
    "mean",
    "max",
]

# Historical/current selector configuration.
BASELINE = {
    "topk": 1024,
    "pooling_kernel": 18,
    "pooling": "mean",
    "baseline": True,
}

MANIFEST_PATH = Path(
    "/Workspace/LMSelfBenchResult-queryfree/"
    "experiments/offline_surprisal_2x2_20260913/"
    "results/raw_qfree_stratified_dev50/"
    "trec/selection_manifest.json"
)

BUCKETS = [
    "0-4k",
    "4-8k",
    "8k+",
]


# =====================================================================
# Helpers
# =====================================================================

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(tmp, path)


def make_grid():
    """
    56 requested grid configurations + 1 historical baseline.
    """

    configs = []

    idx = 0

    for topk in TOPKS:
        for kernel in POOLING_KERNELS:
            for pooling in POOLING_METHODS:

                configs.append({
                    "index": idx,
                    "topk": int(topk),
                    "pooling_kernel": int(kernel),
                    "pooling": str(pooling),
                    "baseline": False,
                })

                idx += 1

    baseline = dict(BASELINE)
    baseline["index"] = idx

    configs.append(baseline)

    if len(configs) != 57:
        raise RuntimeError(
            f"Expected 57 configurations, got {len(configs)}"
        )

    return configs


def config_name(cfg):
    suffix = "_baseline" if cfg["baseline"] else ""

    return (
        f"cfg{cfg['index']:03d}"
        f"_K{cfg['topk']}"
        f"_k{cfg['pooling_kernel']}"
        f"_{cfg['pooling']}"
        f"{suffix}"
    )


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"TREC manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    if manifest.get("dataset") != DATASET:
        raise RuntimeError(
            "Manifest is not the TREC manifest"
        )

    samples = manifest.get("samples", [])

    if len(samples) != 50:
        raise RuntimeError(
            f"Expected 50 TREC samples, got {len(samples)}"
        )

    indices = [
        int(x["sample_index"])
        for x in samples
    ]

    if len(set(indices)) != 50:
        raise RuntimeError(
            "TREC manifest contains duplicate indices"
        )

    return manifest


def config_root(output_root, cfg):
    return (
        Path(output_root)
        / "configs"
        / config_name(cfg)
    )


def sample_path(output_root, cfg, sample_index):
    return (
        config_root(output_root, cfg)
        / "samples"
        / f"{int(sample_index):06d}.json"
    )


# =====================================================================
# Dynamic pooling / TopK
# =====================================================================

def dynamic_pool(score, pooling, kernel):
    """
    Match the pooling semantics used by LAMB / the audited six-arm
    implementation.

    For even kernels such as historical kernel=18, torch pooling gives
    N+1 outputs with padding=k//2. The audited implementation truncates
    the extra final position back to N.
    """

    score = score.reshape(-1)

    if score.numel() == 0:
        return score

    x = (
        score
        .float()
        .reshape(1, 1, -1)
    )

    if pooling == "mean":
        pooled = F.avg_pool1d(
            x,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )

    elif pooling == "max":
        pooled = F.max_pool1d(
            x,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )

    else:
        raise ValueError(
            f"Unknown pooling: {pooling}"
        )

    pooled = pooled.reshape(-1)

    # Official even-kernel alignment.
    if pooled.numel() > score.numel():
        pooled = pooled[:score.numel()]

    if pooled.numel() != score.numel():
        raise RuntimeError(
            f"Pooling length mismatch: "
            f"{pooled.numel()} vs {score.numel()}"
        )

    return pooled


def dynamic_budget_mask(
    score,
    topk,
    pooling,
    kernel,
):
    """
    Dynamic equivalent of the existing
    six.build_budget_mask_from_score().

    Historical semantics:
      - pool Surprisal
      - last VIS_WINDOW positions are always retained
      - TopK is chosen only from earlier positions
      - if sequence is shorter than TopK + recent window,
        everything is retained.
    """

    score = score.reshape(-1)

    n = int(score.numel())

    if n == 0:
        return torch.zeros(
            0,
            dtype=torch.bool,
            device=score.device,
        )

    if n <= int(topk) + VIS_WINDOW:
        return torch.ones(
            n,
            dtype=torch.bool,
            device=score.device,
        )

    pooled = dynamic_pool(
        score,
        pooling=pooling,
        kernel=kernel,
    )

    eligible_n = n - VIS_WINDOW

    eligible_score = pooled[:eligible_n]

    k = min(
        int(topk),
        eligible_n,
    )

    mask = torch.zeros(
        n,
        dtype=torch.bool,
        device=score.device,
    )

    if k > 0:
        selected = torch.topk(
            eligible_score,
            k=k,
            largest=True,
            sorted=False,
        ).indices

        mask[selected] = True

    mask[-VIS_WINDOW:] = True

    expected = min(
        n,
        int(topk) + VIS_WINDOW,
    )

    if int(mask.sum().item()) != expected:
        raise RuntimeError(
            "Dynamic budget mismatch: "
            f"selected={int(mask.sum())}, "
            f"expected={expected}"
        )

    return mask


def baseline_equivalence_check(device):
    """
    Critical safety check.

    Our dynamic implementation MUST exactly reproduce the existing
    audited six-arm builder for K=1024 / mean / kernel=18.
    """

    print()
    print("=" * 80)
    print("BASELINE MASK EQUIVALENCE CHECK")
    print("=" * 80)

    torch.manual_seed(20260913)

    test_lengths = [
        500,
        1056,
        1057,
        2048,
        4096,
        8192,
    ]

    for n in test_lengths:
        score = torch.randn(
            n,
            device=device,
            dtype=torch.float32,
        )

        old_mask = (
            six.build_budget_mask_from_score(
                score
            )
        )

        new_mask = dynamic_budget_mask(
            score,
            topk=1024,
            pooling="mean",
            kernel=18,
        )

        identical = torch.equal(
            old_mask,
            new_mask,
        )

        overlap = int(
            (old_mask & new_mask)
            .sum()
            .item()
        )

        print(
            f"N={n:5d} | "
            f"old={int(old_mask.sum()):4d} | "
            f"new={int(new_mask.sum()):4d} | "
            f"overlap={overlap:4d} | "
            f"identical={identical}"
        )

        if not identical:
            diff = int(
                (old_mask != new_mask)
                .sum()
                .item()
            )

            raise RuntimeError(
                "Dynamic builder does NOT exactly match "
                "the audited baseline builder. "
                f"N={n}, differing positions={diff}. "
                "Sweep aborted."
            )

    print("BASELINE EQUIVALENCE: EXACT")
    print("=" * 80)


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_config(
    output_root,
    cfg,
    manifest,
):
    root = config_root(
        output_root,
        cfg,
    )

    rows = []

    for item in manifest["samples"]:
        idx = int(
            item["sample_index"]
        )

        p = sample_path(
            output_root,
            cfg,
            idx,
        )

        if not p.exists():
            return None

        rows.append(
            json.loads(
                p.read_text(
                    encoding="utf-8"
                )
            )
        )

    predictions = [
        x["prediction"]
        for x in rows
    ]

    answers = [
        x["answers"]
        for x in rows
    ]

    lengths = [
        int(x["raw_length"])
        for x in rows
    ]

    all_classes = rows[0][
        "all_classes"
    ]

    bucket_scores = scorer_e(
        DATASET,
        predictions,
        answers,
        lengths,
        all_classes,
    )

    final_score = float(
        np.mean([
            float(bucket_scores[b])
            for b in BUCKETS
        ])
    )

    diagnostic = np.asarray(
        [
            float(x["diagnostic_score"])
            for x in rows
        ],
        dtype=np.float64,
    )

    result = {
        "config_index":
            int(cfg["index"]),

        "config_name":
            config_name(cfg),

        "topk":
            int(cfg["topk"]),

        "pooling_kernel":
            int(cfg["pooling_kernel"]),

        "pooling":
            cfg["pooling"],

        "baseline":
            bool(cfg["baseline"]),

        "samples":
            len(rows),

        "0-4k":
            float(
                bucket_scores["0-4k"]
            ),

        "4-8k":
            float(
                bucket_scores["4-8k"]
            ),

        "8k+":
            float(
                bucket_scores["8k+"]
            ),

        "final":
            final_score,

        "diagnostic_sample_mean":
            float(
                diagnostic.mean()
            ),
    }

    atomic_json(
        root / "result.json",
        result,
    )

    return result


def aggregate_all(
    output_root,
    configs,
    manifest,
):
    results = []

    incomplete = []

    for cfg in configs:
        result = evaluate_config(
            output_root,
            cfg,
            manifest,
        )

        if result is None:
            incomplete.append(
                config_name(cfg)
            )
        else:
            results.append(result)

    results.sort(
        key=lambda x: (
            -x["final"],
            x["topk"],
            x["pooling_kernel"],
            x["pooling"],
        )
    )

    baseline = next(
        (
            x for x in results
            if x["baseline"]
        ),
        None,
    )

    baseline_score = (
        baseline["final"]
        if baseline is not None
        else None
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result["rank"] = rank

        if baseline_score is not None:
            result["delta_vs_baseline"] = (
                result["final"]
                - baseline_score
            )
        else:
            result["delta_vs_baseline"] = None

    summary = {
        "dataset":
            DATASET,

        "samples":
            50,

        "total_configs":
            len(configs),

        "completed_configs":
            len(results),

        "incomplete_configs":
            incomplete,

        "baseline":
            baseline,

        "ranking":
            results,
    }

    atomic_json(
        Path(output_root)
        / "sweep_summary.json",
        summary,
    )

    csv_path = (
        Path(output_root)
        / "sweep_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "config_index",
            "topk",
            "pooling_kernel",
            "pooling",
            "baseline",
            "0-4k",
            "4-8k",
            "8k+",
            "final",
            "delta_vs_baseline",
        ])

        for x in results:
            writer.writerow([
                x["rank"],
                x["config_index"],
                x["topk"],
                x["pooling_kernel"],
                x["pooling"],
                x["baseline"],
                x["0-4k"],
                x["4-8k"],
                x["8k+"],
                x["final"],
                x["delta_vs_baseline"],
            ])

    print()
    print("=" * 125)
    print(
        "TREC RAW Q-FREE PARAMETER SWEEP — DEV-50"
    )
    print("=" * 125)

    print(
        f"{'Rank':>5s}"
        f"{'ID':>6s}"
        f"{'TopK':>8s}"
        f"{'Kernel':>9s}"
        f"{'Pool':>8s}"
        f"{'0-4k':>10s}"
        f"{'4-8k':>10s}"
        f"{'8k+':>10s}"
        f"{'FINAL':>10s}"
        f"{'Δbase':>10s}"
    )

    print("-" * 125)

    for x in results:
        delta = (
            x["delta_vs_baseline"]
        )

        delta_text = (
            f"{delta:+.2f}"
            if delta is not None
            else "NA"
        )

        marker = (
            " BASE"
            if x["baseline"]
            else ""
        )

        print(
            f"{x['rank']:5d}"
            f"{x['config_index']:6d}"
            f"{x['topk']:8d}"
            f"{x['pooling_kernel']:9d}"
            f"{x['pooling']:>8s}"
            f"{x['0-4k']:10.2f}"
            f"{x['4-8k']:10.2f}"
            f"{x['8k+']:10.2f}"
            f"{x['final']:10.2f}"
            f"{delta_text:>10s}"
            f"{marker}"
        )

    print("-" * 125)

    print(
        f"Completed: {len(results)}/{len(configs)}"
    )

    if incomplete:
        print(
            f"Incomplete: {len(incomplete)}"
        )

    if baseline is not None:
        print(
            "Historical baseline "
            f"K=1024 / mean / k=18: "
            f"{baseline['final']:.2f}"
        )

    if len(results) >= 3:
        print()
        print("TOP-3:")
        for x in results[:3]:
            print(
                f"  #{x['rank']} "
                f"K={x['topk']} "
                f"kernel={x['pooling_kernel']} "
                f"pool={x['pooling']} "
                f"FINAL={x['final']:.2f} "
                f"Δbase={x['delta_vs_baseline']:+.2f}"
            )

        atomic_json(
            Path(output_root)
            / "top3_configs.json",
            results[:3],
        )

    print("=" * 125)

    return summary


# =====================================================================
# Main worker
# =====================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--logit-chunk-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )

    parser.add_argument(
        "--print-grid",
        action="store_true",
    )

    args = parser.parse_args()

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    configs = make_grid()

    if args.print_grid:
        print(
            f"{'ID':>4s} "
            f"{'TopK':>6s} "
            f"{'Kernel':>7s} "
            f"{'Pool':>6s} "
            f"{'Worker(3)':>10s}"
        )

        print("-" * 45)

        for cfg in configs:
            print(
                f"{cfg['index']:4d} "
                f"{cfg['topk']:6d} "
                f"{cfg['pooling_kernel']:7d} "
                f"{cfg['pooling']:>6s} "
                f"{cfg['index'] % 3:10d}"
                + (
                    "  BASELINE"
                    if cfg["baseline"]
                    else ""
                )
            )

        return

    manifest = load_manifest()

    if args.aggregate_only:
        aggregate_all(
            output_root,
            configs,
            manifest,
        )
        return

    if args.num_workers <= 0:
        raise ValueError(
            "--num-workers must be > 0"
        )

    if not (
        0 <= args.worker_id
        < args.num_workers
    ):
        raise ValueError(
            "worker-id must satisfy "
            "0 <= worker-id < num-workers"
        )

    worker_configs = [
        cfg
        for cfg in configs
        if (
            cfg["index"]
            % args.num_workers
        )
        == args.worker_id
    ]

    print()
    print("=" * 100)
    print(
        f"WORKER {args.worker_id}/"
        f"{args.num_workers}"
    )
    print("=" * 100)

    print(
        "Assigned configs:",
        len(worker_configs),
    )

    for cfg in worker_configs:
        print(
            f"  {config_name(cfg)}"
        )

    print("=" * 100)

    disable_progress_bar()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = f"cuda:{args.device}"

    # ---------------------------------------------------------
    # Critical equivalence audit before any real experiment.
    # ---------------------------------------------------------

    baseline_equivalence_check(
        device,
    )

    # ---------------------------------------------------------
    # Existing audited LAMB configuration / Global Channels.
    # ---------------------------------------------------------

    cfg_lamb = six.preflight()

    base.seed_all(
        42 + args.worker_id
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "Fast tokenizer required"
        )

    print(
        f"[worker {args.worker_id}] "
        "Loading Mamba2-1.3B..."
    )

    model = base.MambaLMHeadModel.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
    )

    model.eval()

    controller = base.RoutingController(
        model,
        cfg_lamb,
        device,
    )

    if len(controller.layers) != 48:
        raise RuntimeError(
            f"Expected 48 layers, "
            f"got {len(controller.layers)}"
        )

    # ---------------------------------------------------------
    # Load exact same TREC-E dataset.
    # ---------------------------------------------------------

    ds = load_dataset(
        "THUDM/LongBench",
        "trec_e",
        split="test",
        trust_remote_code=True,
    )

    # ---------------------------------------------------------
    # Save global protocol.
    # ---------------------------------------------------------

    protocol_path = (
        output_root
        / "protocol.json"
    )

    if not protocol_path.exists():
        atomic_json(
            protocol_path,
            {
                "experiment":
                    "TREC Raw Q-Free parameter sweep",

                "dataset":
                    "THUDM/LongBench trec_e",

                "samples":
                    50,

                "manifest":
                    str(MANIFEST_PATH),

                "method":
                    "Raw Context-only Q-Free Surprisal + "
                    "canonical trailing-space trim",

                "fixed": {
                    "model":
                        MODEL_NAME,

                    "vis_window":
                        VIS_WINDOW,

                    "normalization":
                        "none",

                    "question_handling":
                        "Q-Free selector; non-context prompt "
                        "positions pass through",

                    "global_channels":
                        "official LAMB precomputed channels",

                    "generation":
                        "existing six.generate_one",
                },

                "sweep": {
                    "topk":
                        TOPKS,

                    "pooling_kernel":
                        POOLING_KERNELS,

                    "pooling":
                        POOLING_METHODS,

                    "extra_baseline":
                        BASELINE,

                    "total_configs":
                        57,
                },
            },
        )

    # ---------------------------------------------------------
    # Run sample-major.
    #
    # This is much faster than config-major because Context
    # Surprisal is calculated only once per sample per worker.
    # ---------------------------------------------------------

    pbar = tqdm(
        manifest["samples"],
        desc=f"TREC worker {args.worker_id}",
        dynamic_ncols=True,
    )

    for item in pbar:
        idx = int(
            item["sample_index"]
        )

        pending = [
            cfg
            for cfg in worker_configs
            if not sample_path(
                output_root,
                cfg,
                idx,
            ).exists()
        ]

        if not pending:
            continue

        sample = ds[idx]

        template = base.dataset2prompt[
            DATASET
        ]

        (
            prompt,
            (
                ctx_char_start,
                ctx_char_end,
            ),
            field_spans,
        ) = six.render_prompt_and_context_span(
            template,
            sample,
        )

        context_text = str(
            sample["context"]
        )

        if (
            prompt[
                ctx_char_start:
                ctx_char_end
            ]
            != context_text
        ):
            raise RuntimeError(
                f"TREC/{idx}: Context span mismatch"
            )

        # -----------------------------------------------------
        # Canonical historical prompt handling.
        # -----------------------------------------------------

        if prompt.endswith(" "):
            prompt = prompt[:-1]

        if (
            prompt[
                ctx_char_start:
                ctx_char_end
            ]
            != context_text
        ):
            raise RuntimeError(
                f"TREC/{idx}: "
                "trailing-space trim changed Context span"
            )

        # -----------------------------------------------------
        # Full prompt tokenization.
        # -----------------------------------------------------

        prompt_enc = tokenizer(
            prompt,
            return_offsets_mapping=True,
            truncation=False,
        )

        prompt_offsets = (
            prompt_enc[
                "offset_mapping"
            ]
        )

        input_ids = torch.tensor(
            [
                prompt_enc[
                    "input_ids"
                ]
            ],
            dtype=torch.long,
            device=device,
        )

        prompt_ctx_indices = (
            six.token_indices_overlapping_span(
                prompt_offsets,
                ctx_char_start,
                ctx_char_end,
            )
        )

        if not prompt_ctx_indices:
            raise RuntimeError(
                f"TREC/{idx}: "
                "No Context prompt positions"
            )

        # -----------------------------------------------------
        # Context-only raw causal Surprisal.
        # Reuse existing audited function.
        # -----------------------------------------------------

        context_enc = tokenizer(
            context_text,
            return_offsets_mapping=True,
            truncation=False,
        )

        context_ids = torch.tensor(
            [
                context_enc[
                    "input_ids"
                ]
            ],
            dtype=torch.long,
            device=device,
        )

        context_raw = (
            six.compute_full_causal_surprisal(
                model,
                controller,
                context_ids,
                args.logit_chunk_size,
            )
        )

        projected = (
            six.project_context_scores_to_prompt(
                context_raw,
                context_enc[
                    "offset_mapping"
                ],
                prompt_offsets,
                prompt_ctx_indices,
                ctx_char_start,
                len(context_text),
            )
        )

        # -----------------------------------------------------
        # Build all pending masks.
        #
        # Group identical final prompt masks together so that
        # identical configurations only generate once.
        # -----------------------------------------------------

        mask_groups = defaultdict(
            list
        )

        mask_objects = {}

        for cfg in pending:
            local_mask = dynamic_budget_mask(
                projected,
                topk=cfg["topk"],
                pooling=cfg["pooling"],
                kernel=cfg[
                    "pooling_kernel"
                ],
            )

            expected_context = min(
                len(prompt_ctx_indices),
                int(cfg["topk"])
                + VIS_WINDOW,
            )

            if (
                int(local_mask.sum().item())
                != expected_context
            ):
                raise RuntimeError(
                    f"{config_name(cfg)} "
                    f"Context budget mismatch: "
                    f"{int(local_mask.sum())} != "
                    f"{expected_context}"
                )

            prompt_mask = (
                six.context_local_to_prompt_pass_mask(
                    local_mask,
                    prompt_ctx_indices,
                    int(
                        input_ids.shape[1]
                    ),
                    device,
                )
            )

            # Exact binary mask signature.
            signature = (
                prompt_mask
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.uint8,
                )
                .numpy()
                .tobytes()
            )

            mask_groups[
                signature
            ].append(
                (
                    cfg,
                    local_mask,
                )
            )

            mask_objects[
                signature
            ] = prompt_mask

        seed = six.stable_sample_seed(
            DATASET,
            idx,
        )

        unique_masks = len(
            mask_groups
        )

        # -----------------------------------------------------
        # Generate once per unique routing mask.
        # -----------------------------------------------------

        for signature, grouped in (
            mask_groups.items()
        ):
            prompt_mask = (
                mask_objects[
                    signature
                ]
            )

            prediction, score = (
                six.generate_one(
                    model=model,
                    controller=controller,
                    tokenizer=tokenizer,
                    dataset=DATASET,
                    sample=sample,
                    input_ids=input_ids,
                    prompt_mask=prompt_mask,
                    seed=seed,
                )
            )

            for (
                cfg,
                local_mask,
            ) in grouped:

                row = {
                    "dataset":
                        DATASET,

                    "sample_index":
                        idx,

                    "_id":
                        sample.get(
                            "_id"
                        ),

                    "bucket":
                        item.get(
                            "bucket"
                        ),

                    "raw_length":
                        int(
                            sample[
                                "length"
                            ]
                        ),

                    "answers":
                        sample[
                            "answers"
                        ],

                    "all_classes":
                        sample[
                            "all_classes"
                        ],

                    "config": {
                        "index":
                            int(
                                cfg[
                                    "index"
                                ]
                            ),

                        "topk":
                            int(
                                cfg[
                                    "topk"
                                ]
                            ),

                        "pooling_kernel":
                            int(
                                cfg[
                                    "pooling_kernel"
                                ]
                            ),

                        "pooling":
                            cfg[
                                "pooling"
                            ],

                        "baseline":
                            bool(
                                cfg[
                                    "baseline"
                                ]
                            ),
                    },

                    "prediction":
                        prediction,

                    "diagnostic_score":
                        float(
                            score
                        ),

                    "prompt_tokens":
                        int(
                            input_ids.shape[1]
                        ),

                    "context_prompt_tokens":
                        len(
                            prompt_ctx_indices
                        ),

                    "context_selected":
                        int(
                            local_mask
                            .sum()
                            .item()
                        ),

                    "selected_total":
                        int(
                            prompt_mask
                            .sum()
                            .item()
                        ),

                    "generation_shared_with":
                        len(grouped),

                    "field_spans":
                        field_spans,
                }

                p = sample_path(
                    output_root,
                    cfg,
                    idx,
                )

                p.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                atomic_json(
                    p,
                    row,
                )

        pbar.set_postfix(
            pending=len(pending),
            unique_masks=unique_masks,
        )

        del (
            context_ids,
            context_raw,
            projected,
            input_ids,
            mask_groups,
            mask_objects,
        )

        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Score this worker's completed configurations.
    # ---------------------------------------------------------

    print()
    print(
        f"[worker {args.worker_id}] "
        "Evaluating assigned configurations..."
    )

    completed = 0

    for cfg in worker_configs:
        result = evaluate_config(
            output_root,
            cfg,
            manifest,
        )

        if result is not None:
            completed += 1

            print(
                f"{config_name(cfg)} "
                f"=> FINAL "
                f"{result['final']:.2f}"
            )

    print()
    print(
        f"WORKER {args.worker_id} COMPLETE: "
        f"{completed}/{len(worker_configs)} configs"
    )


if __name__ == "__main__":
    main()
