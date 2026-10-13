#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import string
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import disable_progress_bar, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXP = SCRIPT_DIR
BASE_SCRIPT_DIR = SCRIPT_DIR
sys.path.insert(0, str(BASE_SCRIPT_DIR))
import mamba2_runtime as base

MODEL_NAME = 'state-spaces/mamba2-1.3b'
TOKENIZER_NAME = 'EleutherAI/gpt-neox-20b'
CONFIG_PATH = Path(
    os.environ.get(
        "LAMB_ROOT",
        str(SCRIPT_DIR.parent / "third_party/lamb")
    )
) / "remapping_configs/mamba2-1.3b_topk1024_pk9.json"
TOPK = 1024
VIS_WINDOW = 32
POOLING = 'mean'
POOLING_KERNEL = 18
NORM_WINDOW = 65
BOUNDED_CORE = 1024
BOUNDED_LEFT = 128
DP_ALPHA = 0.25
N_LAYERS = 48
BASE_SEED = 42

DATASETS = [
    'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa',
    'gov_report', 'multi_news', 'trec', 'triviaqa', 'samsum',
    'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p',
]

CATEGORIES = {
    'single_doc_qa': ['qasper', 'multifieldqa_en'],
    'multi_doc_qa': ['hotpotqa', '2wikimqa'],
    'summarization': ['gov_report', 'multi_news'],
    'few_shot': ['trec', 'triviaqa', 'samsum'],
    'synthetic': ['passage_count', 'passage_retrieval_en'],
    'code': ['lcc', 'repobench-p'],
}

ARMS = {
    'prompt_canon_s': '01_prompt_canon_s',
    'promptS_ctxTopK_pass': '02_promptS_ctxTopK_pass',
    'qfree_context_s': '03_qfree_context_s',
    'qfree_bounded_s': '04_qfree_bounded_s',
    'qfree_multiscale_s': '05_qfree_multiscale_s',
    'qfree_disagreement_s': '06_qfree_disagreement_s',
}

REQUIRED_BASE_SYMBOLS = [
    'MambaLMHeadModel', 'MODEL_NAME', 'TOKENIZER_NAME', 'dataset2prompt',
    'dataset2maxgenlen', 'RoutingController', 'compute_final_surprisal',
    'compute_bounded_layer_surprisal', 'rolling_mean_std_zscore',
    'official_mean_pool', 'build_official_budget_mask_from_pooled',
    'generate_answer', 'score_prediction', 'seed_all',
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def stable_sample_seed(dataset, sample_index):
    payload = f'{BASE_SEED}|{dataset}|{sample_index}'.encode('utf-8')
    value = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    return value % (2**31 - 1)


def preflight():
    missing = [x for x in REQUIRED_BASE_SYMBOLS if not hasattr(base, x)]
    if missing:
        raise RuntimeError(f'Base script missing required symbols: {missing}')
    if base.MODEL_NAME != MODEL_NAME:
        raise RuntimeError(f'Model mismatch: base={base.MODEL_NAME} expected={MODEL_NAME}')
    if base.TOKENIZER_NAME != TOKENIZER_NAME:
        raise RuntimeError(f'Tokenizer mismatch: base={base.TOKENIZER_NAME} expected={TOKENIZER_NAME}')
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    cfg = read_json(CONFIG_PATH)
    rk = cfg['remap_kwargs']
    checks = {
        'attn_topk': TOPK,
        'vis_window': VIS_WINDOW,
        'pooling': POOLING,
        'pooling_kernel': POOLING_KERNEL,
    }
    for key, expected in checks.items():
        actual = rk.get(key)
        if actual != expected:
            raise RuntimeError(f'Config mismatch: {key}={actual!r}, expected {expected!r}')
    for d in DATASETS:
        if d not in base.dataset2prompt:
            raise RuntimeError(f'Missing dataset2prompt entry: {d}')
        if d not in base.dataset2maxgenlen:
            raise RuntimeError(f'Missing dataset2maxgenlen entry: {d}')
    return cfg


def render_prompt_and_context_span(template, sample):
    formatter = string.Formatter()
    pieces = []
    cursor = 0
    context_spans = []
    field_spans = defaultdict(list)
    for literal, field_name, format_spec, conversion in formatter.parse(template):
        pieces.append(literal)
        cursor += len(literal)
        if field_name is None:
            continue
        value, _ = formatter.get_field(field_name, (), sample)
        if conversion:
            value = formatter.convert_field(value, conversion)
        rendered = formatter.format_field(value, format_spec)
        start = cursor
        pieces.append(rendered)
        cursor += len(rendered)
        end = cursor
        field_spans[field_name].append((start, end))
        if field_name == 'context':
            context_spans.append((start, end))
    prompt = ''.join(pieces)
    expected = template.format(**sample)
    if prompt != expected:
        raise RuntimeError('Structural prompt rendering mismatch')
    if len(context_spans) != 1:
        raise RuntimeError(f'Expected exactly one {{context}} field, found {len(context_spans)}')
    return prompt, context_spans[0], dict(field_spans)


def token_indices_overlapping_span(offsets, start_char, end_char):
    result = []
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if e > start_char and s < end_char:
            result.append(i)
    return result


def project_context_scores_to_prompt(raw_context_scores, context_offsets, prompt_offsets,
                                     prompt_context_indices, prompt_context_char_start,
                                     context_text_length):
    scores = raw_context_scores.detach().float().reshape(-1).cpu().numpy()
    if len(scores) != len(context_offsets):
        raise RuntimeError(
            f'Context score/token mismatch: scores={len(scores)} offsets={len(context_offsets)}'
        )
    valid = []
    for idx, (s, e) in enumerate(context_offsets):
        if s != e:
            valid.append((idx, int(s), int(e)))
    if not valid:
        raise RuntimeError('No valid Context token offsets')

    projected = []
    j = 0
    for pidx in prompt_context_indices:
        ps, pe = prompt_offsets[pidx]
        rs = max(0, int(ps) - int(prompt_context_char_start))
        re = min(int(context_text_length), int(pe) - int(prompt_context_char_start))
        if re <= rs:
            re = min(int(context_text_length), rs + 1)

        while j < len(valid) and valid[j][2] <= rs:
            j += 1
        k = j
        weighted = 0.0
        overlap_total = 0
        while k < len(valid) and valid[k][1] < re:
            cidx, cs, ce = valid[k]
            ov = max(0, min(re, ce) - max(rs, cs))
            if ov > 0:
                weighted += ov * float(scores[cidx])
                overlap_total += ov
            k += 1

        if overlap_total > 0:
            projected.append(weighted / overlap_total)
        else:
            candidates = []
            if j < len(valid):
                candidates.append(valid[j])
            if j > 0:
                candidates.append(valid[j - 1])
            if not candidates:
                raise RuntimeError('Projection failed: no nearest Context token')
            center = 0.5 * (rs + re)
            nearest = min(candidates, key=lambda x: abs(0.5 * (x[1] + x[2]) - center))
            projected.append(float(scores[nearest[0]]))

    out = torch.tensor(projected, dtype=torch.float32, device=raw_context_scores.device)
    if out.numel() != len(prompt_context_indices):
        raise RuntimeError('Projected score length mismatch')
    if not torch.isfinite(out).all():
        raise RuntimeError('Non-finite projected Context scores')
    return out


def build_budget_mask_from_score(score):
    """
    Fixed-budget selector:

        official mean pooling (kernel=18)
            -> EXACT historical Top1024
            -> EXACT recent32

    For an even pooling kernel, symmetric padding may make the pooled
    sequence one element longer than the historical token sequence.
    That extra right-boundary pooled position must NOT be allowed to
    enter the historical TopK, because it aliases the first recent
    token and can reduce the unique budget from 1056 to 1055.

    We therefore align the pooled score back to the historical token
    domain before calling the existing official budget-mask builder.
    No selector score, normalization, ranking rule, TopK, or recent
    window is otherwise changed.
    """
    score = score.reshape(-1)
    n = int(score.numel())

    if n == 0:
        raise RuntimeError('Cannot select from empty sequence')

    if not torch.isfinite(score).all():
        raise RuntimeError('Selector score contains non-finite values')

    # Short sequence: every token is retained.
    if n <= TOPK + VIS_WINDOW:
        return torch.ones(
            n,
            dtype=torch.bool,
            device=score.device,
        )

    history_end = n - VIS_WINDOW
    history = score[:history_end]
    history_len = int(history.numel())

    pooled = base.official_mean_pool(
        history,
        POOLING_KERNEL,
    ).reshape(-1)

    pooled_len_before_alignment = int(
        pooled.numel()
    )

    # kernel=18 is even. The exact prior mean-pooling implementation can
    # produce history_len + 1 positions because of symmetric padding.
    # Drop ONLY the extra right boundary position.
    if pooled_len_before_alignment == history_len + 1:
        pooled = pooled[:history_len]

    elif pooled_len_before_alignment != history_len:
        raise RuntimeError(
            'Unexpected pooled/history length relation: '
            f'history={history_len}, '
            f'pooled={pooled_len_before_alignment}, '
            f'kernel={POOLING_KERNEL}'
        )

    if int(pooled.numel()) != history_len:
        raise RuntimeError(
            'Pooling alignment invariant failed: '
            f'pooled={pooled.numel()} history={history_len}'
        )

    if not torch.isfinite(pooled).all():
        raise RuntimeError(
            'Aligned pooled selector score contains non-finite values'
        )

    mask = base.build_official_budget_mask_from_pooled(
        pooled_score=pooled,
        seq_len=n,
        topk=TOPK,
        vis_window=VIS_WINDOW,
        device=score.device,
    )

    mask = mask.reshape(-1).bool()

    if mask.numel() != n:
        raise RuntimeError(
            f'Mask length mismatch: {mask.numel()} != {n}'
        )

    historical_selected = int(
        mask[:history_end].sum().item()
    )

    recent_selected = int(
        mask[history_end:].sum().item()
    )

    total_selected = int(
        mask.sum().item()
    )

    # Stronger than the old total-count check:
    # verify the two components independently.
    if historical_selected != TOPK:
        raise RuntimeError(
            'Historical budget invariant failed: '
            f'selected={historical_selected}, '
            f'expected={TOPK}, '
            f'n={n}, '
            f'history={history_len}, '
            f'pooled_before_alignment={pooled_len_before_alignment}'
        )

    if recent_selected != VIS_WINDOW:
        raise RuntimeError(
            'Recent-window budget invariant failed: '
            f'selected={recent_selected}, '
            f'expected={VIS_WINDOW}, '
            f'n={n}'
        )

    expected_total = TOPK + VIS_WINDOW

    if total_selected != expected_total:
        raise RuntimeError(
            'Total budget invariant failed: '
            f'selected={total_selected}, '
            f'expected={expected_total}, '
            f'n={n}'
        )

    if not bool(
        mask[-VIS_WINDOW:].all().item()
    ):
        raise RuntimeError(
            'Recent-32 positional invariant failed'
        )

    return mask


def context_local_to_prompt_pass_mask(local_mask, prompt_context_indices, prompt_len, device):
    if local_mask.numel() != len(prompt_context_indices):
        raise RuntimeError('Context local mask/token-index mismatch')
    # Non-context pass-through: instruction/question/separators update Global + Local normally.
    prompt_mask = torch.ones(prompt_len, dtype=torch.bool, device=device)
    ctx_idx = torch.tensor(prompt_context_indices, dtype=torch.long, device=device)
    prompt_mask[ctx_idx] = local_mask.to(device=device, dtype=torch.bool)
    return prompt_mask


def repeat_mask(mask):
    return [mask for _ in range(N_LAYERS)]


def compute_full_causal_surprisal(model, controller, ids, logit_chunk_size):
    controller.vanilla()
    with torch.inference_mode():
        hidden = model.backbone(ids)
        raw = base.compute_final_surprisal(
            model=model,
            hidden=hidden,
            input_ids=ids,
            chunk_size=logit_chunk_size,
        )
    del hidden
    raw = raw.reshape(-1)
    if raw.numel() != ids.shape[1]:
        raise RuntimeError(f'Surprisal length mismatch: {raw.numel()} != {ids.shape[1]}')
    if not torch.isfinite(raw).all():
        raise RuntimeError('Non-finite causal Surprisal')
    return raw


def compute_bounded_context_surprisal(model, controller, context_ids, logit_chunk_size):
    bounded = base.compute_bounded_layer_surprisal(
        model=model,
        controller=controller,
        input_ids=context_ids,
        anchor_layers=[48],
        core_size=BOUNDED_CORE,
        left_context=BOUNDED_LEFT,
        logit_chunk_size=logit_chunk_size,
    )
    raw = bounded[0].reshape(-1)
    if raw.numel() != context_ids.shape[1]:
        raise RuntimeError(f'Bounded Surprisal length mismatch: {raw.numel()} != {context_ids.shape[1]}')
    if not torch.isfinite(raw).all():
        raise RuntimeError('Non-finite bounded Surprisal')
    return raw


def generate_one(model, controller, tokenizer, dataset, sample, input_ids, prompt_mask, seed):
    base.seed_all(seed)
    controller.external_masks(repeat_mask(prompt_mask))
    pred = base.generate_answer(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_gen=int(base.dataset2maxgenlen[dataset]),
    )
    score = base.score_prediction(
        dataset=dataset,
        prediction=pred,
        answers=sample['answers'],
        all_classes=sample['all_classes'],
    )
    return pred, float(score)


def arm_sample_path(root, arm, dataset, idx):
    return Path(root) / ARMS[arm] / 'generation' / dataset / 'samples' / f'{idx:06d}.json'


def ensure_manifests(root, selected_arms, protocol):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    root_protocol = root / 'protocol.json'
    if root_protocol.exists():
        old_root = read_json(root_protocol)
        if old_root.get('protocol_signature') != protocol['protocol_signature']:
            raise RuntimeError(
                f'Protocol mismatch in {root_protocol}. Use a NEW output root; do not mix runs.'
            )
    else:
        atomic_json(root_protocol, protocol)
    for arm in selected_arms:
        arm_dir = root / ARMS[arm]
        arm_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = arm_dir / 'manifest.json'
        manifest = {
            'arm': arm,
            'folder': ARMS[arm],
            'protocol_signature': protocol['protocol_signature'],
            'method': protocol['arm_definitions'][arm],
        }
        if manifest_path.exists():
            old = read_json(manifest_path)
            if old.get('protocol_signature') != protocol['protocol_signature']:
                raise RuntimeError(
                    f'Protocol mismatch in {manifest_path}. Use a NEW output root; do not mix runs.'
                )
        else:
            atomic_json(manifest_path, manifest)


def read_completed_rows(root, arm, dataset):
    sample_dir = Path(root) / ARMS[arm] / 'generation' / dataset / 'samples'
    if not sample_dir.exists():
        return []
    rows = []
    for path in sorted(sample_dir.glob('*.json')):
        rows.append(read_json(path))
    return rows


def aggregate_arm(root, arm, targets):
    arm_dir = Path(root) / ARMS[arm]
    rows_out = []
    per_dataset = {}
    for dataset in DATASETS:
        rows = read_completed_rows(root, arm, dataset)
        scores = [float(r['score']) for r in rows]
        target = int(targets.get(dataset, 0))
        n = len(scores)
        mean = float(np.mean(scores)) if scores else None
        formal_50 = bool(target >= 50 and n >= 50)
        complete = target > 0 and n >= target
        row = {
            'dataset': dataset,
            'n_complete': n,
            'target_n': target,
            'completion_pct': (100.0 * n / target) if target else 0.0,
            'formal_50_reached': bool(formal_50),
            'complete': bool(complete),
            'mean_score': mean,
        }
        rows_out.append(row)
        per_dataset[dataset] = row

        # Convenience JSONL rebuilt atomically from per-sample files.
        gen_dir = arm_dir / 'generation'
        gen_dir.mkdir(parents=True, exist_ok=True)
        jsonl_tmp = gen_dir / f'{dataset}.jsonl.tmp'
        with open(jsonl_tmp, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        os.replace(jsonl_tmp, gen_dir / f'{dataset}.jsonl')

    summary_csv = arm_dir / 'summary.csv'
    with open(summary_csv.with_suffix('.csv.tmp'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    os.replace(summary_csv.with_suffix('.csv.tmp'), summary_csv)

    available_means = [x['mean_score'] for x in rows_out if x['mean_score'] is not None]
    full_macro = float(np.mean(available_means)) if available_means else None
    macro_status = 'FULL' if all(x['complete'] for x in rows_out) else 'PARTIAL'
    formal_50_status = all(x['formal_50_reached'] for x in rows_out if x['target_n'] > 0)

    category_summary = {}
    for category, ds_list in CATEGORIES.items():
        vals = [per_dataset[d]['mean_score'] for d in ds_list if per_dataset[d]['mean_score'] is not None]
        category_summary[category] = {
            'datasets_available': len(vals),
            'datasets_total': len(ds_list),
            'macro': float(np.mean(vals)) if vals else None,
            'status': 'FULL' if len(vals) == len(ds_list) and all(per_dataset[d]['complete'] for d in ds_list) else 'PARTIAL',
        }

    summary = {
        'arm': arm,
        'folder': ARMS[arm],
        'status': macro_status,
        'all_datasets_reached_50': bool(formal_50_status),
        'dataset_macro_over_available': full_macro,
        'datasets': per_dataset,
        'categories': category_summary,
        'warning': (
            'PARTIAL summaries are progress reports only. Mechanism conclusions require >=50 samples '
            'for every included dataset; final benchmark claims require complete LongBench-E.'
        ),
    }
    atomic_json(arm_dir / 'summary.json', summary)
    return summary


def aggregate_all(root, selected_arms, targets):
    summaries = {}
    for arm in selected_arms:
        summaries[arm] = aggregate_arm(root, arm, targets)
    root_status = {
        'arms': {
            arm: {
                'folder': ARMS[arm],
                'status': summaries[arm]['status'],
                'all_datasets_reached_50': summaries[arm]['all_datasets_reached_50'],
                'dataset_macro_over_available': summaries[arm]['dataset_macro_over_available'],
            }
            for arm in selected_arms
        }
    }
    atomic_json(Path(root) / 'STATUS.json', root_status)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--arms', default=','.join(ARMS.keys()))
    parser.add_argument('--first-pass', type=int, default=50,
                        help='Run this many samples on every dataset before continuing to full test.')
    parser.add_argument('--limit-per-dataset', type=int, default=0,
                        help='0 = full dataset. Positive value is smoke/debug only.')
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--logit-chunk-size', type=int, default=64)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()

    disable_progress_bar()
    cfg = preflight()

    selected_arms = [x.strip() for x in args.arms.split(',') if x.strip()]
    unknown = [x for x in selected_arms if x not in ARMS]
    if unknown:
        raise ValueError(f'Unknown arms: {unknown}')
    if not selected_arms:
        raise ValueError('No arms selected')

    script_hash = sha256_file(Path(__file__).resolve())
    base_hash = sha256_file(Path(base.__file__).resolve())
    protocol_core = {
        'model': MODEL_NAME,
        'tokenizer': TOKENIZER_NAME,
        'config': str(CONFIG_PATH),
        'topk': TOPK,
        'vis_window': VIS_WINDOW,
        'pooling': POOLING,
        'pooling_kernel': POOLING_KERNEL,
        'norm_window': NORM_WINDOW,
        'bounded_core': BOUNDED_CORE,
        'bounded_left_context': BOUNDED_LEFT,
        'dp_alpha': DP_ALPHA,
        'n_layers': N_LAYERS,
        'base_seed': BASE_SEED,
        'datasets': DATASETS,
        'script_sha256': script_hash,
        'base_script_sha256': base_hash,
        'config_remap_kwargs': cfg['remap_kwargs'],
    }
    signature = hashlib.sha256(
        json.dumps(protocol_core, sort_keys=True, ensure_ascii=False).encode('utf-8')
    ).hexdigest()
    protocol = {
        **protocol_core,
        'protocol_signature': signature,
        'arm_definitions': {
            'prompt_canon_s': (
                'Full prompt causal Surprisal -> rolling local mean/std z-score (window=65) -> '
                'official LAMB mean pooling (kernel=18) -> historical Top1024 + prompt recent32. '
                'This reproduces the prior Canon-S definition.'
            ),
            'promptS_ctxTopK_pass': (
                'Surprisal and local normalization computed on the FULL prompt (Question included). '
                'Only Context token scores enter Context Top1024 + Context recent32 selection. '
                'All non-Context prompt tokens pass through Global+Local normally.'
            ),
            'qfree_context_s': (
                'Raw Context-only causal Surprisal -> alignment onto Context positions of the official prompt -> '
                'rolling local mean/std z-score -> official mean pooling -> Context Top1024 + Context recent32. '
                'All non-Context prompt tokens pass through Global+Local normally.'
            ),
            'qfree_bounded_s': (
                'Context-only bounded Surprisal using core=1024,left=128, layer-48 anchor -> prompt Context alignment -> '
                'rolling local mean/std z-score -> official mean pooling -> Context Top1024 + Context recent32; '
                'non-Context prompt tokens pass through Global+Local.'
            ),
            'qfree_multiscale_s': (
                'Context-only full and bounded Surprisal are independently aligned and independently rolling-z normalized; '
                'token score = 0.5*z_full + 0.5*z_bounded; then official mean pooling -> Context Top1024 + recent32; '
                'non-Context prompt tokens pass through Global+Local. No tuning or learned parameters.'
            ),
            'qfree_disagreement_s': (
                'Same two independently normalized Context-only scales; token score = '
                '0.5*(z_full+z_bounded) - 0.25*abs(z_full-z_bounded); official mean pooling -> '
                'Context Top1024 + recent32; non-Context prompt tokens pass through Global+Local. '
                'alpha=0.25 is the fixed exploratory value proposed before this full run and is not swept.'
            ),
        },
        'important_invariants': [
            'No Vanilla Mamba, LongMamba, or LAMB generation is run by this script.',
            'No prompt or Context truncation is introduced by this script.',
            'Question/instruction tokens do not compete for Context TopK in query-free/pass-through arms.',
            'Question/instruction tokens still update Global and Local channels in pass-through arms.',
            'All Context selectors use fixed Top1024 + recent32 and official mean pooling kernel18.',
            'Canonical and bounded Surprisal use the exact prior base implementation functions.',
            'No quantile calibration, clipping, layer-band selector, LAMB fusion, coverage constraint, or retriever is added.',
        ],
    }
    ensure_manifests(args.output_root, selected_arms, protocol)

    targets_path = Path(args.output_root) / 'dataset_targets.json'
    targets = read_json(targets_path) if targets_path.exists() else {}

    if args.aggregate_only:
        if not targets:
            raise RuntimeError('dataset_targets.json not found; run at least once before --aggregate-only')
        summaries = aggregate_all(args.output_root, selected_arms, targets)
        for arm, s in summaries.items():
            print(f'{arm:30s} status={s["status"]:7s} macro={s["dataset_macro_over_available"]}')
        return

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = f'cuda:{args.device}'
    base.seed_all(BASE_SEED)

    print('Loading tokenizer...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if not tokenizer.is_fast:
        raise RuntimeError('Fast tokenizer required for structural token alignment')

    print('Loading Mamba2-1.3B...', flush=True)
    model = base.MambaLMHeadModel.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    controller = base.RoutingController(model, cfg, device)
    if len(controller.layers) != N_LAYERS:
        raise RuntimeError(f'Expected {N_LAYERS} layers, found {len(controller.layers)}')

    # First load only to establish immutable target counts.
    for dataset in DATASETS:
        ds = load_dataset('THUDM/LongBench', f'{dataset}_e', split='test', trust_remote_code=True)
        target = len(ds) if args.limit_per_dataset <= 0 else min(len(ds), args.limit_per_dataset)
        if dataset in targets and int(targets[dataset]) != int(target):
            raise RuntimeError(f'Target count changed for {dataset}: old={targets[dataset]} new={target}')
        targets[dataset] = int(target)
        del ds
    atomic_json(targets_path, targets)

    first_pass = max(0, int(args.first_pass))
    stages = []
    if first_pass > 0:
        stages.append(('FIRST_PASS', 0, first_pass))
    stages.append(('REMAINDER', first_pass if first_pass > 0 else 0, None))

    completed_since_checkpoint = 0

    for stage_name, stage_start, stage_stop in stages:
        print('\n' + '=' * 120)
        print(f'STAGE: {stage_name}')
        print('=' * 120)

        for dataset in DATASETS:
            target = int(targets[dataset])
            start = min(stage_start, target)
            stop = target if stage_stop is None else min(stage_stop, target)
            if start >= stop:
                continue

            ds = load_dataset('THUDM/LongBench', f'{dataset}_e', split='test', trust_remote_code=True)
            if len(ds) < target:
                raise RuntimeError(f'Dataset {dataset} shorter than recorded target')

            pbar = tqdm(range(start, stop), desc=f'{stage_name}:{dataset}', dynamic_ncols=True)
            for idx in pbar:
                missing_arms = [
                    arm for arm in selected_arms
                    if not arm_sample_path(args.output_root, arm, dataset, idx).exists()
                ]
                if not missing_arms:
                    continue

                sample = ds[idx]
                template = base.dataset2prompt[dataset]
                prompt, (ctx_char_start, ctx_char_end), field_spans = render_prompt_and_context_span(template, sample)
                context_text = str(sample['context'])
                if prompt[ctx_char_start:ctx_char_end] != context_text:
                    raise RuntimeError(f'{dataset}/{idx}: Context span content mismatch')

                prompt_enc = tokenizer(prompt, return_offsets_mapping=True, truncation=False)
                prompt_ids_list = prompt_enc['input_ids']
                prompt_offsets = prompt_enc['offset_mapping']
                input_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
                prompt_len = int(input_ids.shape[1])

                prompt_ctx_indices = token_indices_overlapping_span(
                    prompt_offsets, ctx_char_start, ctx_char_end
                )
                if not prompt_ctx_indices:
                    raise RuntimeError(f'{dataset}/{idx}: no prompt Context tokens')

                context_enc = tokenizer(context_text, return_offsets_mapping=True, truncation=False)
                context_ids_list = context_enc['input_ids']
                context_offsets = context_enc['offset_mapping']
                context_ids = torch.tensor([context_ids_list], dtype=torch.long, device=device)
                if context_ids.shape[1] == 0:
                    raise RuntimeError(f'{dataset}/{idx}: empty tokenized Context')

                need_prompt = any(a in missing_arms for a in ['prompt_canon_s', 'promptS_ctxTopK_pass'])
                need_full_ctx = any(a in missing_arms for a in ['qfree_context_s', 'qfree_multiscale_s', 'qfree_disagreement_s'])
                need_bounded = any(a in missing_arms for a in ['qfree_bounded_s', 'qfree_multiscale_s', 'qfree_disagreement_s'])

                prompt_raw = prompt_z = None
                full_ctx_projected = full_ctx_z = None
                bounded_projected = bounded_z = None

                if need_prompt:
                    prompt_raw = compute_full_causal_surprisal(
                        model, controller, input_ids, args.logit_chunk_size
                    )
                    prompt_z = base.rolling_mean_std_zscore(prompt_raw, NORM_WINDOW).reshape(-1)
                    if prompt_z.numel() != prompt_len or not torch.isfinite(prompt_z).all():
                        raise RuntimeError(f'{dataset}/{idx}: invalid prompt normalized Surprisal')

                if need_full_ctx:
                    full_ctx_raw = compute_full_causal_surprisal(
                        model, controller, context_ids, args.logit_chunk_size
                    )
                    full_ctx_projected = project_context_scores_to_prompt(
                        full_ctx_raw, context_offsets, prompt_offsets, prompt_ctx_indices,
                        ctx_char_start, len(context_text)
                    )
                    full_ctx_z = base.rolling_mean_std_zscore(
                        full_ctx_projected, NORM_WINDOW
                    ).reshape(-1)
                    del full_ctx_raw

                if need_bounded:
                    bounded_raw = compute_bounded_context_surprisal(
                        model, controller, context_ids, args.logit_chunk_size
                    )
                    bounded_projected = project_context_scores_to_prompt(
                        bounded_raw, context_offsets, prompt_offsets, prompt_ctx_indices,
                        ctx_char_start, len(context_text)
                    )
                    bounded_z = base.rolling_mean_std_zscore(
                        bounded_projected, NORM_WINDOW
                    ).reshape(-1)
                    del bounded_raw

                masks = {}
                local_selected_counts = {}

                if 'prompt_canon_s' in missing_arms:
                    m = build_budget_mask_from_score(prompt_z)
                    masks['prompt_canon_s'] = m
                    local_selected_counts['prompt_canon_s'] = int(m.sum().item())

                if 'promptS_ctxTopK_pass' in missing_arms:
                    ctx_tensor = torch.tensor(prompt_ctx_indices, dtype=torch.long, device=device)
                    ctx_score = prompt_z[ctx_tensor]
                    local = build_budget_mask_from_score(ctx_score)
                    masks['promptS_ctxTopK_pass'] = context_local_to_prompt_pass_mask(
                        local, prompt_ctx_indices, prompt_len, device
                    )
                    local_selected_counts['promptS_ctxTopK_pass'] = int(local.sum().item())

                if 'qfree_context_s' in missing_arms:
                    local = build_budget_mask_from_score(full_ctx_z)
                    masks['qfree_context_s'] = context_local_to_prompt_pass_mask(
                        local, prompt_ctx_indices, prompt_len, device
                    )
                    local_selected_counts['qfree_context_s'] = int(local.sum().item())

                if 'qfree_bounded_s' in missing_arms:
                    local = build_budget_mask_from_score(bounded_z)
                    masks['qfree_bounded_s'] = context_local_to_prompt_pass_mask(
                        local, prompt_ctx_indices, prompt_len, device
                    )
                    local_selected_counts['qfree_bounded_s'] = int(local.sum().item())

                if 'qfree_multiscale_s' in missing_arms:
                    multi_score = 0.5 * full_ctx_z + 0.5 * bounded_z
                    local = build_budget_mask_from_score(multi_score)
                    masks['qfree_multiscale_s'] = context_local_to_prompt_pass_mask(
                        local, prompt_ctx_indices, prompt_len, device
                    )
                    local_selected_counts['qfree_multiscale_s'] = int(local.sum().item())

                if 'qfree_disagreement_s' in missing_arms:
                    dp_score = 0.5 * (full_ctx_z + bounded_z) - DP_ALPHA * torch.abs(full_ctx_z - bounded_z)
                    local = build_budget_mask_from_score(dp_score)
                    masks['qfree_disagreement_s'] = context_local_to_prompt_pass_mask(
                        local, prompt_ctx_indices, prompt_len, device
                    )
                    local_selected_counts['qfree_disagreement_s'] = int(local.sum().item())

                expected_context_selected = min(len(prompt_ctx_indices), TOPK + VIS_WINDOW)
                noncontext_count = prompt_len - len(prompt_ctx_indices)
                for arm, mask in masks.items():
                    if mask.numel() != prompt_len:
                        raise RuntimeError(f'{dataset}/{idx}/{arm}: prompt mask length mismatch')
                    if arm != 'prompt_canon_s':
                        ctx_idx_tensor = torch.tensor(prompt_ctx_indices, dtype=torch.long, device=device)
                        nonctx = torch.ones(prompt_len, dtype=torch.bool, device=device)
                        nonctx[ctx_idx_tensor] = False
                        if not bool(mask[nonctx].all().item()):
                            raise RuntimeError(f'{dataset}/{idx}/{arm}: non-Context pass-through invariant failed')
                        if local_selected_counts[arm] != expected_context_selected:
                            raise RuntimeError(
                                f'{dataset}/{idx}/{arm}: Context budget mismatch '
                                f'{local_selected_counts[arm]} != {expected_context_selected}'
                            )

                seed = stable_sample_seed(dataset, idx)
                for arm in missing_arms:
                    pred, score = generate_one(
                        model, controller, tokenizer, dataset, sample, input_ids, masks[arm], seed
                    )
                    record = {
                        'dataset': dataset,
                        'sample_index': int(idx),
                        '_id': sample.get('_id'),
                        'arm': arm,
                        'prediction': pred,
                        'score': score,
                        'answers': sample['answers'],
                        'diagnostics': {
                            'prompt_tokens': prompt_len,
                            'context_prompt_tokens': len(prompt_ctx_indices),
                            'context_only_tokens': int(context_ids.shape[1]),
                            'context_selected_tokens': (
                                local_selected_counts.get(arm)
                                if arm != 'prompt_canon_s' else None
                            ),
                            'total_global_write_tokens_in_prefill': int(masks[arm].sum().item()),
                            'noncontext_prompt_tokens': noncontext_count,
                            'noncontext_pass_through': arm != 'prompt_canon_s',
                            'field_spans': field_spans,
                            'sample_seed': seed,
                        },
                        'protocol_signature': signature,
                    }
                    atomic_json(arm_sample_path(args.output_root, arm, dataset, idx), record)

                completed_since_checkpoint += 1
                if completed_since_checkpoint >= max(1, args.checkpoint_every):
                    aggregate_all(args.output_root, selected_arms, targets)
                    completed_since_checkpoint = 0

                del input_ids, context_ids, masks
                if prompt_raw is not None:
                    del prompt_raw, prompt_z
                if full_ctx_projected is not None:
                    del full_ctx_projected, full_ctx_z
                if bounded_projected is not None:
                    del bounded_projected, bounded_z
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del ds
            aggregate_all(args.output_root, selected_arms, targets)

        if stage_name == 'FIRST_PASS':
            summaries = aggregate_all(args.output_root, selected_arms, targets)
            first_pass_ok = all(
                all(
                    s['datasets'][d]['n_complete'] >= min(first_pass, targets[d])
                    for d in DATASETS
                )
                for s in summaries.values()
            )
            if first_pass_ok:
                atomic_json(
                    Path(args.output_root) / f'FIRST_PASS_{first_pass}_COMPLETE.json',
                    {
                        'complete': True,
                        'first_pass': first_pass,
                        'note': 'Every selected arm reached the first-pass sample target on every LongBench-E dataset.',
                    },
                )

    summaries = aggregate_all(args.output_root, selected_arms, targets)
    print('\n' + '=' * 120)
    print('FINAL / CURRENT STATUS')
    print('=' * 120)
    for arm in selected_arms:
        s = summaries[arm]
        print(
            f'{ARMS[arm]:32s} status={s["status"]:7s} '
            f'all>=50={str(s["all_datasets_reached_50"]):5s} '
            f'macro={s["dataset_macro_over_available"]}'
        )
    print(f'\nSaved under: {args.output_root}')


if __name__ == '__main__':
    main()
