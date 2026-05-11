#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_COLUMNS = ("is_vpn", "label", "vpn", "target", "class")
RAW_FEATURES = (
    "flow_duration",
    "fwd_num_pkts",
    "bwd_num_pkts",
    "fwd_sum_pkt_len",
    "bwd_sum_pkt_len",
    "fwd_min_pkt_len",
    "fwd_mean_pkt_len",
    "fwd_std_pkt_len",
    "fwd_max_pkt_len",
    "bwd_min_pkt_len",
    "bwd_mean_pkt_len",
    "bwd_std_pkt_len",
    "bwd_max_pkt_len",
    "fwd_min_iat",
    "fwd_mean_iat",
    "fwd_std_iat",
    "fwd_max_iat",
    "bwd_min_iat",
    "bwd_mean_iat",
    "bwd_std_iat",
    "bwd_max_iat",
    "fwd_num_syn_flags",
    "fwd_num_ack_flags",
    "fwd_num_fin_flags",
    "fwd_num_rst_flags",
    "fwd_num_psh_flags",
    "fwd_num_urg_flags",
    "bwd_num_syn_flags",
    "bwd_num_ack_flags",
    "bwd_num_fin_flags",
    "bwd_num_rst_flags",
    "bwd_num_psh_flags",
    "bwd_num_urg_flags",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tiny NN model using engineered features (ratios/log interactions)."
    )
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--label-column", type=str, default="")
    parser.add_argument("--hidden-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--l2", type=float, default=0.00001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--target-recall", type=float, default=0.0)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--submissions-path",
        type=Path,
        default=PROJECT_ROOT / "submission.py",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "trained_vpn_model.txt",
    )
    parser.add_argument("--update-submission", action="store_true")
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_to_int(value: object) -> int:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "vpn"}:
            return 1
        if v in {"0", "false", "no", "nonvpn", "non-vpn"}:
            return 0
    return 1 if _safe_float(value) >= 0.5 else 0


def _detect_label_column(fieldnames: list[str], requested: str) -> str:
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"Label column '{requested}' not found.")
        return requested
    for c in DEFAULT_LABEL_COLUMNS:
        if c in fieldnames:
            return c
    raise ValueError("Could not detect label column.")


def _sigmoid(x: float) -> float:
    x = max(-35.0, min(35.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _engineer_row(raw: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {name: raw.get(name, 0.0) for name in RAW_FEATURES}

    flow_duration = out["flow_duration"]
    fwd_num_pkts = out["fwd_num_pkts"]
    bwd_num_pkts = out["bwd_num_pkts"]
    fwd_sum_pkt_len = out["fwd_sum_pkt_len"]
    bwd_sum_pkt_len = out["bwd_sum_pkt_len"]

    total_pkts = fwd_num_pkts + bwd_num_pkts
    total_bytes = fwd_sum_pkt_len + bwd_sum_pkt_len

    duration_ms = flow_duration / 1000.0 if flow_duration > 10_000 else flow_duration
    duration_s = max(duration_ms / 1000.0, 1e-6)

    avg_pkt_len = _safe_div(total_bytes, total_pkts)
    pkts_per_sec = _safe_div(total_pkts, duration_s)
    bytes_per_sec = _safe_div(total_bytes, duration_s)

    pkt_balance = _safe_div(abs(fwd_num_pkts - bwd_num_pkts), total_pkts)
    byte_balance = _safe_div(abs(fwd_sum_pkt_len - bwd_sum_pkt_len), total_bytes)
    bi_pkt_ratio = _safe_div(
        min(fwd_num_pkts, bwd_num_pkts), max(fwd_num_pkts, bwd_num_pkts)
    )
    bi_byte_ratio = _safe_div(
        min(fwd_sum_pkt_len, bwd_sum_pkt_len), max(fwd_sum_pkt_len, bwd_sum_pkt_len)
    )

    out.update(
        {
            "total_pkts": total_pkts,
            "total_bytes": total_bytes,
            "duration_ms": duration_ms,
            "pkts_per_sec": pkts_per_sec,
            "bytes_per_sec": bytes_per_sec,
            "avg_pkt_len": avg_pkt_len,
            "pkt_balance": pkt_balance,
            "byte_balance": byte_balance,
            "bi_pkt_ratio": bi_pkt_ratio,
            "bi_byte_ratio": bi_byte_ratio,
            "ack_balance": _safe_div(
                min(out["fwd_num_ack_flags"], out["bwd_num_ack_flags"]),
                max(out["fwd_num_ack_flags"], out["bwd_num_ack_flags"]),
            ),
            "syn_total": out["fwd_num_syn_flags"] + out["bwd_num_syn_flags"],
            "rst_total": out["fwd_num_rst_flags"] + out["bwd_num_rst_flags"],
            "iat_mean_ratio": _safe_div(
                min(out["fwd_mean_iat"], out["bwd_mean_iat"]),
                max(out["fwd_mean_iat"], out["bwd_mean_iat"]),
            ),
            "iat_std_total": out["fwd_std_iat"] + out["bwd_std_iat"],
            "pkt_byte_interaction": bi_pkt_ratio * bi_byte_ratio,
            "flow_compactness": _safe_div(total_bytes, duration_ms + 1.0),
            "small_flow": 1.0 if total_pkts <= 12.0 else 0.0,
            "tiny_bytes": 1.0 if total_bytes <= 4000.0 else 0.0,
        }
    )

    for name in RAW_FEATURES:
        value = out[name]
        out[f"log1p_{name}"] = math.log1p(value) if value > 0.0 else 0.0

    out["log1p_total_pkts"] = math.log1p(total_pkts) if total_pkts > 0.0 else 0.0
    out["log1p_total_bytes"] = math.log1p(total_bytes) if total_bytes > 0.0 else 0.0
    out["log1p_pkts_per_sec"] = math.log1p(pkts_per_sec) if pkts_per_sec > 0.0 else 0.0
    out["log1p_bytes_per_sec"] = (
        math.log1p(bytes_per_sec) if bytes_per_sec > 0.0 else 0.0
    )

    return out


def _load_csv(
    csv_path: Path, label_column: str, max_rows: int
) -> tuple[list[list[float]], list[int], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw_rows: list[dict[str, float]] = []
    labels: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        y_col = _detect_label_column(fieldnames, label_column)

        missing = [name for name in RAW_FEATURES if name not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing required raw features: {', '.join(missing)}")

        for row in reader:
            raw = {name: _safe_float(row.get(name), 0.0) for name in RAW_FEATURES}
            raw_rows.append(raw)
            labels.append(_label_to_int(row.get(y_col)))
            if max_rows > 0 and len(raw_rows) >= max_rows:
                break

    if not raw_rows:
        raise RuntimeError("No rows loaded from CSV.")

    engineered_rows = [_engineer_row(raw) for raw in raw_rows]
    feature_names = list(engineered_rows[0].keys())
    X = [[row[name] for name in feature_names] for row in engineered_rows]
    return X, labels, feature_names


def _stats(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(X)
    d = len(X[0])
    means = [0.0] * d
    stds = [0.0] * d

    for row in X:
        for i, v in enumerate(row):
            means[i] += v
    means = [v / n for v in means]

    for row in X:
        for i, v in enumerate(row):
            diff = v - means[i]
            stds[i] += diff * diff
    stds = [math.sqrt(v / n) if v > 0.0 else 1.0 for v in stds]
    stds = [s if s > 0.0 else 1.0 for s in stds]
    return means, stds


def _normalize(
    X: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    out: list[list[float]] = []
    for row in X:
        out.append([(v - means[i]) / stds[i] for i, v in enumerate(row)])
    return out


def _init_params(
    input_dim: int, hidden_size: int, seed: int
) -> tuple[list[list[float]], list[float], list[float], float]:
    rng = random.Random(seed)
    scale1 = 1.0 / math.sqrt(max(1, input_dim))
    W1 = [
        [(rng.random() * 2.0 - 1.0) * scale1 for _ in range(hidden_size)]
        for _ in range(input_dim)
    ]
    b1 = [0.0 for _ in range(hidden_size)]
    W2 = [
        (rng.random() * 2.0 - 1.0) * (1.0 / math.sqrt(max(1, hidden_size)))
        for _ in range(hidden_size)
    ]
    b2 = 0.0
    return W1, b1, W2, b2


def _forward(
    x: list[float], W1: list[list[float]], b1: list[float], W2: list[float], b2: float
) -> tuple[list[float], float, float]:
    h: list[float] = []
    for j in range(len(b1)):
        z = b1[j]
        for i, xv in enumerate(x):
            z += xv * W1[i][j]
        h.append(math.tanh(z))

    logit = b2
    for j, hv in enumerate(h):
        logit += hv * W2[j]
    p = _sigmoid(logit)
    return h, logit, p


def _evaluate(
    X: list[list[float]],
    y: list[int],
    threshold: float,
    W1: list[list[float]],
    b1: list[float],
    W2: list[float],
    b2: float,
) -> tuple[float, float, float, int, int, int, int]:
    tp = fp = tn = fn = 0
    for xv, label in zip(X, y):
        _, _, p = _forward(xv, W1, b1, W2, b2)
        pred = 1 if p >= threshold else 0
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif (not pred) and (not label):
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    return f1, precision, recall, tp, fp, tn, fn


def _best_threshold(
    X: list[list[float]],
    y: list[int],
    W1: list[list[float]],
    b1: list[float],
    W2: list[float],
    b2: float,
    target_recall: float = 0.0,
    min_precision: float = 0.0,
) -> tuple[float, tuple[float, float, float, int, int, int, int]]:
    target_recall = max(0.0, min(1.0, target_recall))
    min_precision = max(0.0, min(1.0, min_precision))
    best_t = 0.5
    best = _evaluate(X, y, best_t, W1, b1, W2, b2)
    best_ok = (best[2] >= target_recall) and (best[1] >= min_precision)
    for i in range(1, 100):
        t = i / 100.0
        stats = _evaluate(X, y, t, W1, b1, W2, b2)
        ok = (stats[2] >= target_recall) and (stats[1] >= min_precision)
        if ok and not best_ok:
            best_t, best, best_ok = t, stats, True
            continue
        if ok and best_ok and stats[0] > best[0]:
            best_t, best = t, stats
            continue
        if (not best_ok) and (not ok) and stats[0] > best[0]:
            best_t, best = t, stats
    return best_t, best


def _threshold_curve(
    X: list[list[float]],
    y: list[int],
    W1: list[list[float]],
    b1: list[float],
    W2: list[float],
    b2: float,
) -> list[tuple[float, tuple[float, float, float, int, int, int, int]]]:
    curve: list[tuple[float, tuple[float, float, float, int, int, int, int]]] = []
    for i in range(1, 100):
        t = i / 100.0
        curve.append((t, _evaluate(X, y, t, W1, b1, W2, b2)))
    return curve


def _train(
    X_train: list[list[float]],
    y_train: list[int],
    X_val: list[list[float]],
    y_val: list[int],
    hidden_size: int,
    epochs: int,
    lr: float,
    l2: float,
    dropout: float,
    batch_size: int,
    seed: int,
    target_recall: float,
    min_precision: float,
) -> tuple[
    list[list[float]],
    list[float],
    list[float],
    float,
    float,
    tuple[float, float, float, int, int, int, int],
]:
    W1, b1, W2, b2 = _init_params(len(X_train[0]), hidden_size, seed)
    rng = random.Random(seed + 13)
    dropout = max(0.0, min(0.8, dropout))
    keep_prob = 1.0 - dropout if dropout > 0.0 else 1.0

    n = len(X_train)
    for _ in range(max(1, epochs)):
        idx = list(range(n))
        rng.shuffle(idx)

        for start in range(0, n, max(1, batch_size)):
            batch_idx = idx[start : start + max(1, batch_size)]

            gW1 = [[0.0 for _ in range(hidden_size)] for _ in range(len(W1))]
            gb1 = [0.0 for _ in range(hidden_size)]
            gW2 = [0.0 for _ in range(hidden_size)]
            gb2 = 0.0

            for bi in batch_idx:
                x = X_train[bi]
                y = float(y_train[bi])
                h, _, p = _forward(x, W1, b1, W2, b2)
                if dropout > 0.0:
                    # Inverted dropout: keeps activation scale stable.
                    mask = [
                        (1.0 / keep_prob) if (rng.random() < keep_prob) else 0.0
                        for _ in range(hidden_size)
                    ]
                    h_train = [hv * mask[j] for j, hv in enumerate(h)]
                else:
                    mask = [1.0 for _ in range(hidden_size)]
                    h_train = h

                dlogit = p - y
                gb2 += dlogit
                for j in range(hidden_size):
                    gW2[j] += dlogit * h_train[j]

                for j in range(hidden_size):
                    dh = dlogit * W2[j] * mask[j]
                    dz = dh * (1.0 - h[j] * h[j])
                    gb1[j] += dz
                    for i, xv in enumerate(x):
                        gW1[i][j] += dz * xv

            scale = 1.0 / max(1, len(batch_idx))
            for j in range(hidden_size):
                W2[j] -= lr * (gW2[j] * scale + l2 * W2[j])
            b2 -= lr * (gb2 * scale)

            for j in range(hidden_size):
                b1[j] -= lr * (gb1[j] * scale)
            for i in range(len(W1)):
                for j in range(hidden_size):
                    W1[i][j] -= lr * (gW1[i][j] * scale + l2 * W1[i][j])

    threshold, stats = _best_threshold(
        X_val,
        y_val,
        W1,
        b1,
        W2,
        b2,
        target_recall=target_recall,
        min_precision=min_precision,
    )
    return W1, b1, W2, b2, threshold, stats


def _format_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.12f}" for v in values) + "]"


def _format_w1(w1: list[list[float]]) -> str:
    rows = ["["]
    for row in w1:
        rows.append("    " + _format_list(row) + ",")
    rows.append("]")
    return "\n".join(rows)


def _format_float_dict(names: list[str], values: list[float]) -> str:
    lines = ["{"]
    for n, v in zip(names, values):
        lines.append(f'    "{n}": {v:.12f},')
    lines.append("}")
    return "\n".join(lines)


def _format_blob(values: list[float]) -> str:
    return '"""\n' + " ".join(f"{v:.12f}" for v in values) + '\n"""'


def _replace_first(
    source: str, patterns: list[str], replacement: str
) -> tuple[str, bool]:
    for pattern in patterns:
        updated, count = re.subn(
            pattern, replacement, source, count=1, flags=re.MULTILINE
        )
        if count == 1:
            return updated, True
    return source, False


def update_submission(
    path: Path,
    feature_names: list[str],
    means: list[float],
    stds: list[float],
    hidden_size: int,
    w1: list[list[float]],
    b1: list[float],
    w2: list[float],
    b2: float,
    threshold: float,
) -> None:
    source = path.read_text(encoding="utf-8")
    updated = source

    updated, ok = _replace_first(
        updated, [r"VPN_THRESHOLD = [^\n]+"], f"VPN_THRESHOLD = {threshold:.2f}"
    )
    if not ok:
        raise RuntimeError("Could not update VPN_THRESHOLD.")

    updated, ok = _replace_first(
        updated,
        [r"FEATURE_ORDER = \(\n(?:    .*\n)*?\)"],
        "FEATURE_ORDER = (\n" + "\n".join(f'    "{n}",' for n in feature_names) + "\n)",
    )
    if not ok:
        raise RuntimeError("Could not update FEATURE_ORDER.")

    updated, ok = _replace_first(
        updated,
        [
            r"MEANS_BLOB = \"\"\"\n(?:.*\n)*?\"\"\"",
            r"FEATURE_MEANS = \{\n(?:    .*\n)*?\}",
        ],
        "MEANS_BLOB = " + _format_blob(means),
    )
    if not ok:
        raise RuntimeError("Could not update MEANS_BLOB/FEATURE_MEANS.")

    updated, ok = _replace_first(
        updated,
        [
            r"STDS_BLOB = \"\"\"\n(?:.*\n)*?\"\"\"",
            r"FEATURE_STDS = \{\n(?:    .*\n)*?\}",
        ],
        "STDS_BLOB = " + _format_blob(stds),
    )
    if not ok:
        raise RuntimeError("Could not update STDS_BLOB/FEATURE_STDS.")

    updated, ok = _replace_first(
        updated, [r"NN_HIDDEN_SIZE = [^\n]+"], f"NN_HIDDEN_SIZE = {hidden_size}"
    )
    if not ok:
        raise RuntimeError("Could not update NN_HIDDEN_SIZE.")

    updated, ok = _replace_first(
        updated,
        [r"NN_B1_BLOB = \"\"\"\n(?:.*\n)*?\"\"\"", r"NN_B1 = \[[^\]]*\]"],
        "NN_B1_BLOB = " + _format_blob(b1),
    )
    if not ok:
        raise RuntimeError("Could not update NN_B1.")

    w1_flat: list[float] = []
    for row in w1:
        w1_flat.extend(row)
    updated, ok = _replace_first(
        updated,
        [r"NN_W1_BLOB = \"\"\"\n(?:.*\n)*?\"\"\"", r"NN_W1 = \[(?:.|\n)*?\](?=\nNN_B2 =)"],
        "NN_W1_BLOB = " + _format_blob(w1_flat),
    )
    if not ok:
        raise RuntimeError("Could not update NN_W1.")

    updated, ok = _replace_first(updated, [r"NN_B2 = [^\n]+"], f"NN_B2 = {b2:.12f}")
    if not ok:
        raise RuntimeError("Could not update NN_B2.")

    updated, ok = _replace_first(
        updated,
        [r"NN_W2_BLOB = \"\"\"\n(?:.*\n)*?\"\"\"", r"NN_W2 = \[[^\]]*\]"],
        "NN_W2_BLOB = " + _format_blob(w2),
    )
    if not ok:
        raise RuntimeError("Could not update NN_W2.")

    path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    X, y, feature_names = _load_csv(args.csv_path, args.label_column, args.max_rows)

    pairs = list(zip(X, y))
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    val_n = int(len(pairs) * max(0.0, min(0.5, args.validation_split)))
    val_pairs = pairs[:val_n] if val_n > 0 else pairs
    train_pairs = pairs[val_n:] if val_n > 0 else pairs

    X_train_raw = [x for x, _ in train_pairs]
    y_train = [yy for _, yy in train_pairs]
    X_val_raw = [x for x, _ in val_pairs]
    y_val = [yy for _, yy in val_pairs]

    means, stds = _stats(X_train_raw)
    X_train = _normalize(X_train_raw, means, stds)
    X_val = _normalize(X_val_raw, means, stds)

    w1, b1, w2, b2, threshold, stats = _train(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        l2=args.l2,
        dropout=args.dropout,
        batch_size=args.batch_size,
        seed=args.seed,
        target_recall=args.target_recall,
        min_precision=args.min_precision,
    )
    curve = _threshold_curve(X_val, y_val, w1, b1, w2, b2)
    top_curve = sorted(curve, key=lambda item: item[1][0], reverse=True)[:8]
    curve_lines = [
        (
            f"t={t:.2f} f1={s[0]:.6f} p={s[1]:.6f} r={s[2]:.6f} "
            f"tp={s[3]} fp={s[4]} tn={s[5]} fn={s[6]}"
        )
        for t, s in top_curve
    ]

    report = "\n".join(
        [
            f"rows={len(X)}",
            f"train_rows={len(X_train)}",
            f"val_rows={len(X_val)}",
            f"features={len(feature_names)}",
            f"hidden_size={args.hidden_size}",
            f"dropout={args.dropout:.4f}",
            f"target_recall={max(0.0, min(1.0, args.target_recall)):.2f}",
            f"min_precision={max(0.0, min(1.0, args.min_precision)):.2f}",
            f"best_threshold={threshold:.2f}",
            f"val_f1={stats[0]:.6f}",
            f"val_precision={stats[1]:.6f}",
            f"val_recall={stats[2]:.6f}",
            f"val_tp={stats[3]} val_fp={stats[4]} val_tn={stats[5]} val_fn={stats[6]}",
            "",
            "# top_threshold_candidates",
            *curve_lines,
        ]
    )

    if args.update_submission:
        update_submission(
            args.submissions_path,
            feature_names,
            means,
            stds,
            args.hidden_size,
            w1,
            b1,
            w2,
            b2,
            threshold,
        )
        commit_target = (
            args.submissions_path.parent / "src" / "commit" / "submissions.py"
        )
        if commit_target.exists():
            shutil.copyfile(args.submissions_path, commit_target)

    args.output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"saved_report={args.output_path}")
    if args.update_submission:
        print(f"updated_submission={args.submissions_path}")


if __name__ == "__main__":
    main()
