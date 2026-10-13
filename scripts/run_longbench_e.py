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
from datasets import disable_progress_bar, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAMB_ROOT = Path(os.environ.get(
    "LAMB_ROOT", str(PROJECT_ROOT / "third_party/lamb")
))
ARCHIVE_SCRIPTS = PROJECT_ROOT / "super_mamba"
SWEEP_SCRIPTS = PROJECT_ROOT / "super_mamba"

sys.path.insert(0, str(LAMB_ROOT))
sys.path.insert(0, str(ARCHIVE_SCRIPTS))
sys.path.insert(0, str(SWEEP_SCRIPTS))

import surprisal_pipeline as six
import routing as dynamic

base = six.base

from tasks.longbench import scorer_e


MODEL_NAME = "state-spaces/mamba2-1.3b"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"

SELECTION_PATH = (
    PROJECT_ROOT / "configs/category_frozen_full_selection.json"
)

DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

BUCKETS = [
    "0-4k",
    "4-8k",
    "8k+",
]


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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


def config_name(cfg):
    suffix = "_baseline" if cfg["baseline"] else ""

    return (
        f"K{cfg['topk']}"
        f"_k{cfg['pooling_kernel']}"
        f"_{cfg['pooling']}"
        f"{suffix}"
    )


def sample_path(
    root,
    dataset,
    cfg,
    idx,
):
    return (
        Path(root)
        / dataset
        / config_name(cfg)
        / "samples"
        / f"{idx:06d}.json"
    )


def evaluate(
    root,
    dataset,
    cfg,
    ds,
):
    rows = []

    for idx in range(len(ds)):
        p = sample_path(
            root,
            dataset,
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

    scores = scorer_e(
        dataset,
        predictions,
        answers,
        lengths,
        all_classes,
    )

    final = float(
        np.mean([
            float(scores[b])
            for b in BUCKETS
        ])
    )

    return {
        "dataset": dataset,
        "topk": cfg["topk"],
        "pooling_kernel":
            cfg["pooling_kernel"],
        "pooling": cfg["pooling"],
        "baseline": cfg["baseline"],
        "dev50_score":
            cfg["dev50_score"],
        "n": len(ds),
        "0-4k":
            float(scores["0-4k"]),
        "4-8k":
            float(scores["4-8k"]),
        "8k+":
            float(scores["8k+"]),
        "final": final,
    }


def aggregate(root, selection):
    results = []

    for dataset in DATASETS:
        ds = load_dataset(
            "THUDM/LongBench",
            f"{dataset}_e",
            split="test",
            trust_remote_code=True,
        )

        for cfg in selection[
            "datasets"
        ][dataset]:

            r = evaluate(
                root,
                dataset,
                cfg,
                ds,
            )

            if r is None:
                print(
                    "INCOMPLETE:",
                    dataset,
                    config_name(cfg),
                )
                return

            results.append(r)

    atomic_json(
        Path(root) / "full_summary.json",
        results,
    )

    csv_path = (
        Path(root)
        / "full_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "topk",
                "pooling_kernel",
                "pooling",
                "baseline",
                "dev50_score",
                "n",
                "0-4k",
                "4-8k",
                "8k+",
                "final",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 120)
    print("PER-DATASET SELECTED FULL VALIDATION")
    print("=" * 120)

    for dataset in DATASETS:
        subset = [
            r
            for r in results
            if r["dataset"] == dataset
        ]

        subset.sort(
            key=lambda x: -x["final"]
        )

        print()
        print(dataset)

        for rank, r in enumerate(
            subset,
            1,
        ):
            tag = (
                " BASE"
                if r["baseline"]
                else ""
            )

            print(
                f"  {rank}. "
                f"K={r['topk']:4d} "
                f"k={r['pooling_kernel']:2d} "
                f"{r['pooling']:4s} "
                f"DEV50={r['dev50_score']:6.2f} "
                f"FULL={r['final']:6.2f}"
                f"{tag}"
            )

    print()
    print("=" * 120)


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

    args = parser.parse_args()

    selection = json.loads(
        SELECTION_PATH.read_text(
            encoding="utf-8"
        )
    )

    root = Path(
        args.output_root
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.aggregate_only:
        aggregate(
            root,
            selection,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = f"cuda:{args.device}"

    dynamic.baseline_equivalence_check(
        device
    )

    disable_progress_bar()
    cfg_lamb = six.preflight()
    base.seed_all(42)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )

    model = (
        base.MambaLMHeadModel
        .from_pretrained(
            MODEL_NAME,
            device=device,
            dtype=torch.bfloat16,
        )
    )

    model.eval()

    controller = base.RoutingController(
        model,
        cfg_lamb,
        device,
    )

    global_ordinal = 0

    for dataset in DATASETS:
        print()
        print("#" * 100)
        print(
            f"FULL worker "
            f"{args.worker_id}: {dataset}"
        )
        print("#" * 100)

        ds = load_dataset(
            "THUDM/LongBench",
            f"{dataset}_e",
            split="test",
            trust_remote_code=True,
        )

        configs = selection[
            "datasets"
        ][dataset]

        local_indices = []

        for idx in range(len(ds)):
            worker = (
                global_ordinal
                % args.num_workers
            )

            if worker == args.worker_id:
                local_indices.append(idx)

            global_ordinal += 1

        pbar = tqdm(
            local_indices,
            desc=(
                f"{dataset} FULL "
                f"worker {args.worker_id}"
            ),
            dynamic_ncols=True,
        )

        for idx in pbar:
            pending = [
                cfg
                for cfg in configs
                if not sample_path(
                    root,
                    dataset,
                    cfg,
                    idx,
                ).exists()
            ]

            if not pending:
                continue

            sample = ds[idx]

            template = (
                base.dataset2prompt[
                    dataset
                ]
            )

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

            if prompt.endswith(" "):
                prompt = prompt[:-1]

            prompt_enc = tokenizer(
                prompt,
                return_offsets_mapping=True,
                truncation=False,
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

            prompt_offsets = (
                prompt_enc[
                    "offset_mapping"
                ]
            )

            prompt_ctx_indices = (
                six.token_indices_overlapping_span(
                    prompt_offsets,
                    ctx_char_start,
                    ctx_char_end,
                )
            )

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

            groups = defaultdict(list)
            masks = {}

            for cfg in pending:
                local_mask = (
                    dynamic.dynamic_budget_mask(
                        projected,
                        topk=cfg["topk"],
                        pooling=cfg["pooling"],
                        kernel=cfg[
                            "pooling_kernel"
                        ],
                    )
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

                signature = (
                    prompt_mask
                    .detach()
                    .cpu()
                    .to(torch.uint8)
                    .numpy()
                    .tobytes()
                )

                groups[signature].append(
                    (
                        cfg,
                        int(
                            local_mask
                            .sum()
                            .item()
                        ),
                    )
                )

                masks[
                    signature
                ] = prompt_mask

            seed = six.stable_sample_seed(
                dataset,
                idx,
            )

            for (
                signature,
                grouped,
            ) in groups.items():

                prediction, score = (
                    six.generate_one(
                        model=model,
                        controller=controller,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        sample=sample,
                        input_ids=input_ids,
                        prompt_mask=masks[
                            signature
                        ],
                        seed=seed,
                    )
                )

                for (
                    cfg,
                    context_selected,
                ) in grouped:

                    row = {
                        "dataset":
                            dataset,
                        "sample_index":
                            idx,
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
                        "prediction":
                            prediction,
                        "diagnostic_score":
                            float(score),
                        "config":
                            cfg,
                        "context_selected":
                            context_selected,
                        "shared_generation":
                            len(grouped),
                        "field_spans":
                            field_spans,
                    }

                    p = sample_path(
                        root,
                        dataset,
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
                configs=len(pending),
                unique_masks=len(groups),
            )

            del (
                input_ids,
                context_ids,
                context_raw,
                projected,
                groups,
                masks,
            )

            torch.cuda.empty_cache()

    print(
        f"FULL WORKER "
        f"{args.worker_id} COMPLETE"
    )


if __name__ == "__main__":
    main()
