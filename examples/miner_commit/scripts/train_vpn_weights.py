#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_COLUMNS = ("is_vpn", "label", "vpn", "target", "class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hybrid logistic + heuristic VPN model and update submission.py."
    )
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--label-column", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-logistic-weight", type=float, default=0.90)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--feature",
        action="append",
        dest="features",
        help="Optional base feature name(s). If omitted, uses all numeric columns except label.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=PROJECT_ROOT / "trained_vpn_weights.txt"
    )
    parser.add_argument(
        "--submissions-path", type=Path, default=PROJECT_ROOT / "submission.py"
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


def _safe_int_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "vpn", "yes"}:
            return 1
        if normalized in {"0", "false", "nonvpn", "non-vpn", "no"}:
            return 0
    return 1 if _safe_float(value) >= 0.5 else 0


def _detect_label_column(fieldnames: list[str], requested: str) -> str:
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"Label column '{requested}' not found in CSV header.")
        return requested
    for candidate in DEFAULT_LABEL_COLUMNS:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"Unable to detect label column. Tried: {', '.join(DEFAULT_LABEL_COLUMNS)}"
    )


def load_rows(
    csv_path: Path, label_column: str, requested_features: list[str] | None
) -> tuple[list[dict[str, float]], list[int], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: list[dict[str, float]] = []
    labels: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        resolved_label = _detect_label_column(fieldnames, label_column)

        if requested_features:
            missing = [name for name in requested_features if name not in fieldnames]
            if missing:
                raise ValueError(
                    f"Requested features not found in CSV header: {', '.join(missing)}"
                )
            base_feature_names = requested_features
        else:
            base_feature_names = [name for name in fieldnames if name != resolved_label]

        for raw_row in reader:
            rows.append(
                {
                    name: _safe_float(raw_row.get(name), 0.0)
                    for name in base_feature_names
                }
            )
            labels.append(_safe_int_label(raw_row.get(resolved_label)))

    if not rows:
        raise RuntimeError("Training CSV produced no rows.")

    return rows, labels, base_feature_names


def _feature_value(name: str, raw: dict[str, float], cache: dict[str, float]) -> float:
    if name in cache:
        return cache[name]
    if name.startswith("log1p__"):
        base = name[len("log1p__") :]
        value = _feature_value(base, raw, cache)
        computed = math.log1p(max(0.0, value))
    elif name.startswith("ratio__"):
        left, right = name[len("ratio__") :].split("__", 1)
        a = _feature_value(left, raw, cache)
        b = _feature_value(right, raw, cache)
        computed = a / (abs(b) + 1e-6)
    elif name.startswith("diff__"):
        left, right = name[len("diff__") :].split("__", 1)
        computed = _feature_value(left, raw, cache) - _feature_value(right, raw, cache)
    else:
        computed = float(raw.get(name, 0.0))
    cache[name] = computed
    return computed


def build_feature_order(base_features: list[str]) -> list[str]:
    extras: list[str] = []
    for name in (
        "flow_duration",
        "fwd_sum_pkt_len",
        "bwd_sum_pkt_len",
        "fwd_mean_iat",
        "bwd_mean_iat",
    ):
        if name in base_features:
            extras.append(f"log1p__{name}")

    pairs = [
        ("fwd_sum_pkt_len", "bwd_sum_pkt_len"),
        ("fwd_num_pkts", "bwd_num_pkts"),
        ("fwd_mean_iat", "bwd_mean_iat"),
    ]
    for left, right in pairs:
        if left in base_features and right in base_features:
            extras.append(f"ratio__{left}__{right}")
            extras.append(f"diff__{left}__{right}")

    seen: set[str] = set()
    ordered: list[str] = []
    for name in [*base_features, *extras]:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def engineer_rows(
    raw_rows: list[dict[str, float]], feature_order: list[str]
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for raw in raw_rows:
        cache: dict[str, float] = {}
        for name in feature_order:
            _feature_value(name, raw, cache)
        out.append({name: cache.get(name, 0.0) for name in feature_order})
    return out


def _column_stats(
    rows: list[dict[str, float]], feature_names: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    count = max(len(rows), 1)
    for feature in feature_names:
        values = [float(row.get(feature, 0.0)) for row in rows]
        mean = sum(values) / count
        variance = sum((v - mean) ** 2 for v in values) / count
        std = math.sqrt(variance) or 1.0
        means[feature] = mean
        stds[feature] = std
    return means, stds


def _normalized_matrix(
    rows: list[dict[str, float]],
    feature_names: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> list[list[float]]:
    return [
        [
            (float(row.get(feature, 0.0)) - means[feature]) / stds[feature]
            for feature in feature_names
        ]
        for row in rows
    ]


def _sigmoid(value: float) -> float:
    clamped = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-clamped))


def train_logistic_regression(
    X: list[list[float]], y: list[int], *, learning_rate: float, epochs: int, l2: float
) -> tuple[float, list[float]]:
    feature_count = len(X[0]) if X else 0
    bias = 0.0
    weights = [0.0] * feature_count
    sample_count = max(len(X), 1)

    for _ in range(max(1, epochs)):
        grad_bias = 0.0
        grad_weights = [0.0] * feature_count
        for row, label in zip(X, y):
            linear = bias + sum(weight * value for weight, value in zip(weights, row))
            prediction = _sigmoid(linear)
            error = prediction - float(label)
            grad_bias += error
            for index, value in enumerate(row):
                grad_weights[index] += error * value

        bias -= learning_rate * (grad_bias / sample_count)
        for index in range(feature_count):
            regularized = (grad_weights[index] / sample_count) + l2 * weights[index]
            weights[index] -= learning_rate * regularized

    return bias, weights


def _predict_logistic_probability(
    x_norm: list[float], bias: float, weights: list[float]
) -> float:
    linear = bias + sum(weight * value for weight, value in zip(weights, x_norm))
    return _sigmoid(linear)


def _class_means(
    rows: list[dict[str, float]], labels: list[int], feature_names: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    pos = [row for row, y in zip(rows, labels) if y == 1]
    neg = [row for row, y in zip(rows, labels) if y == 0]
    pos_means: dict[str, float] = {}
    neg_means: dict[str, float] = {}
    for feature in feature_names:
        pos_vals = [float(row.get(feature, 0.0)) for row in pos] or [0.0]
        neg_vals = [float(row.get(feature, 0.0)) for row in neg] or [0.0]
        pos_means[feature] = sum(pos_vals) / len(pos_vals)
        neg_means[feature] = sum(neg_vals) / len(neg_vals)
    return pos_means, neg_means


def build_ranked_rules(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
    stds: dict[str, float],
) -> list[tuple[str, dict[str, float]]]:
    pos_means, neg_means = _class_means(rows, labels, feature_names)
    ranked = sorted(
        feature_names,
        key=lambda f: abs(pos_means[f] - neg_means[f]) / max(stds.get(f, 1.0), 1e-9),
        reverse=True,
    )
    out: list[tuple[str, dict[str, float]]] = []
    for f in ranked:
        diff = pos_means[f] - neg_means[f]
        out.append(
            (
                f,
                {
                    "direction": 1.0 if diff >= 0 else -1.0,
                    "midpoint": (pos_means[f] + neg_means[f]) / 2.0,
                    "scale": max(stds.get(f, 1.0), 1e-9),
                    "weight": abs(diff) / max(stds.get(f, 1.0), 1e-9),
                },
            )
        )
    return out


def _heuristic_probability(
    row: dict[str, float], rules: dict[str, dict[str, float]]
) -> float:
    total = 0.0
    total_weight = 0.0
    for name, rule in rules.items():
        value = float(row.get(name, 0.0))
        direction = rule["direction"]
        midpoint = rule["midpoint"]
        scale = max(rule["scale"], 1e-9)
        weight = max(rule["weight"], 0.0)
        signal = direction * ((value - midpoint) / scale)
        total += weight * _sigmoid(signal)
        total_weight += weight
    if total_weight <= 0.0:
        return 0.5
    return total / total_weight


def _f1_at_threshold(
    probabilities: list[float], labels: list[int], threshold: float
) -> tuple[float, float, float, int, int, int, int]:
    tp = fp = tn = fn = 0
    for prob, label in zip(probabilities, labels):
        prediction = 1 if prob >= threshold else 0
        if prediction and label:
            tp += 1
        elif prediction and not label:
            fp += 1
        elif (not prediction) and (not label):
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1, precision, recall, tp, fp, tn, fn


def _best_threshold(
    probabilities: list[float], labels: list[int]
) -> tuple[float, tuple[float, float, float, int, int, int, int]]:
    best_threshold = 0.5
    best_stats = _f1_at_threshold(probabilities, labels, best_threshold)
    for i in range(1, 100):
        threshold = i / 100.0
        stats = _f1_at_threshold(probabilities, labels, threshold)
        if stats[0] > best_stats[0]:
            best_threshold = threshold
            best_stats = stats
    return best_threshold, best_stats


def _replace_block(source: str, start_marker: str, end_marker: str, body: str) -> str:
    start = source.find(start_marker)
    end = source.find(end_marker)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            f"Could not find marker block: {start_marker} .. {end_marker}"
        )
    head = source[: start + len(start_marker)]
    tail = source[end:]
    return f"{head}\n{body}\n{tail}"


def _format_tuple_block(feature_names: list[str]) -> str:
    lines = ["FEATURE_ORDER = ("]
    for name in feature_names:
        lines.append(f'    "{name}",')
    lines.append(")")
    return "\n".join(lines)


def _format_flat_dict(
    var_name: str, values: dict[str, float], feature_names: list[str]
) -> str:
    lines = [f"{var_name} = {{"]
    for key in feature_names:
        lines.append(f'    "{key}": {values[key]:.12f},')
    lines.append("}")
    return "\n".join(lines)


def _format_weights_dict(feature_names: list[str], weights: list[float]) -> str:
    lines = ["MODEL_WEIGHTS = {"]
    for name, weight in zip(feature_names, weights):
        lines.append(f'    "{name}": {weight:.12f},')
    lines.append("}")
    return "\n".join(lines)


def _format_rules_dict(rules: dict[str, dict[str, float]]) -> str:
    lines = ["HEURISTIC_RULES = {"]
    for name, rule in rules.items():
        lines.append(
            f'    "{name}": {{"direction": {rule["direction"]:.12f}, "midpoint": {rule["midpoint"]:.12f}, "scale": {rule["scale"]:.12f}, "weight": {rule["weight"]:.12f}}},'
        )
    lines.append("}")
    return "\n".join(lines)


def update_submission_parameters(
    submissions_path: Path,
    feature_names: list[str],
    *,
    threshold: float,
    ensemble_logistic_weight: float,
    bias: float,
    weights: list[float],
    means: dict[str, float],
    stds: dict[str, float],
    rules: dict[str, dict[str, float]],
) -> None:
    source = submissions_path.read_text(encoding="utf-8")
    source = source.replace(
        "VPN_THRESHOLD = 0.50", f"VPN_THRESHOLD = {threshold:.2f}", 1
    )
    source = source.replace(
        "ENSEMBLE_LOGISTIC_WEIGHT = 0.90",
        f"ENSEMBLE_LOGISTIC_WEIGHT = {max(0.0, min(1.0, ensemble_logistic_weight)):.2f}",
        1,
    )
    source = source.replace("MODEL_BIAS = 0.0", f"MODEL_BIAS = {bias:.12f}", 1)

    source = _replace_block(
        source,
        "# BEGIN_FEATURE_ORDER",
        "# END_FEATURE_ORDER",
        _format_tuple_block(feature_names),
    )
    source = _replace_block(
        source,
        "# BEGIN_MODEL_WEIGHTS",
        "# END_MODEL_WEIGHTS",
        _format_weights_dict(feature_names, weights),
    )
    source = _replace_block(
        source,
        "# BEGIN_FEATURE_MEANS",
        "# END_FEATURE_MEANS",
        _format_flat_dict("FEATURE_MEANS", means, feature_names),
    )
    source = _replace_block(
        source,
        "# BEGIN_FEATURE_STDS",
        "# END_FEATURE_STDS",
        _format_flat_dict("FEATURE_STDS", stds, feature_names),
    )
    source = _replace_block(
        source,
        "# BEGIN_HEURISTIC_RULES",
        "# END_HEURISTIC_RULES",
        _format_rules_dict(rules),
    )
    submissions_path.write_text(source, encoding="utf-8")


def main() -> None:
    args = parse_args()

    raw_rows, labels, base_features = load_rows(
        args.csv_path, args.label_column, list(args.features) if args.features else None
    )
    feature_names = build_feature_order(base_features)
    rows = engineer_rows(raw_rows, feature_names)

    combined = list(zip(rows, labels))
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    split = max(0.0, min(0.5, args.validation_split))
    val_count = int(len(combined) * split)
    train_rows = combined[val_count:]
    val_rows = combined[:val_count] if val_count > 0 else combined

    train_dicts = [row for row, _ in train_rows]
    train_labels = [label for _, label in train_rows]
    val_dicts = [row for row, _ in val_rows]
    val_labels = [label for _, label in val_rows]

    means, stds = _column_stats(train_dicts, feature_names)
    x_train = _normalized_matrix(train_dicts, feature_names, means, stds)
    x_val = _normalized_matrix(val_dicts, feature_names, means, stds)

    ranked_rules = build_ranked_rules(train_dicts, train_labels, feature_names, stds)

    if args.sweep:
        epoch_options = [80, 120, 180]
        l2_options = [0.0001, 0.0005, 0.001]
        weight_options = [0.7, 0.8, 0.9, 1.0]
        rule_options = [8, 12, 16]
    else:
        epoch_options = [args.epochs]
        l2_options = [args.l2]
        weight_options = [max(0.0, min(1.0, args.ensemble_logistic_weight))]
        rule_options = [12]

    best: dict[str, object] = {
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "threshold": 0.5,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "bias": 0.0,
        "weights": [],
        "rules": {},
        "epochs": args.epochs,
        "l2": args.l2,
        "ensemble_weight": args.ensemble_logistic_weight,
    }

    for epochs in epoch_options:
        for l2 in l2_options:
            bias, weights = train_logistic_regression(
                x_train,
                train_labels,
                learning_rate=args.learning_rate,
                epochs=epochs,
                l2=l2,
            )
            logistic_probs = [
                _predict_logistic_probability(row, bias, weights) for row in x_val
            ]

            for rule_count in rule_options:
                selected_rules = dict(
                    ranked_rules[: max(1, min(rule_count, len(ranked_rules)))]
                )
                heuristic_probs = [
                    _heuristic_probability(row, selected_rules) for row in val_dicts
                ]

                for w in weight_options:
                    val_probs = [
                        (w * p_log) + ((1.0 - w) * p_heu)
                        for p_log, p_heu in zip(logistic_probs, heuristic_probs)
                    ]
                    threshold, stats = _best_threshold(val_probs, val_labels)
                    f1, precision, recall, tp, fp, tn, fn = stats
                    if f1 > float(best["f1"]):
                        best.update(
                            {
                                "f1": f1,
                                "precision": precision,
                                "recall": recall,
                                "threshold": threshold,
                                "tp": tp,
                                "fp": fp,
                                "tn": tn,
                                "fn": fn,
                                "bias": bias,
                                "weights": weights,
                                "rules": selected_rules,
                                "epochs": epochs,
                                "l2": l2,
                                "ensemble_weight": w,
                            }
                        )

    report = "\n".join(
        [
            f"rows={len(rows)}",
            f"train_rows={len(train_rows)}",
            f"val_rows={len(val_rows)}",
            f"features={','.join(feature_names)}",
            "model=hybrid_logistic_heuristic",
            f"sweep_enabled={args.sweep}",
            f"best_epochs={int(best['epochs'])}",
            f"best_l2={float(best['l2']):.6f}",
            f"ensemble_logistic_weight={float(best['ensemble_weight']):.2f}",
            f"val_best_threshold={float(best['threshold']):.2f}",
            f"val_f1={float(best['f1']):.6f}",
            f"val_precision={float(best['precision']):.6f}",
            f"val_recall={float(best['recall']):.6f}",
            f"val_tp={int(best['tp'])} val_fp={int(best['fp'])} val_tn={int(best['tn'])} val_fn={int(best['fn'])}",
            "",
            "# MODEL_WEIGHTS",
            _format_weights_dict(feature_names, list(best["weights"])),
            "",
            "# FEATURE_MEANS",
            _format_flat_dict("FEATURE_MEANS", means, feature_names),
            "",
            "# FEATURE_STDS",
            _format_flat_dict("FEATURE_STDS", stds, feature_names),
            "",
            "# HEURISTIC_RULES",
            _format_rules_dict(dict(best["rules"])),
            "",
        ]
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")

    if args.update_submission:
        update_submission_parameters(
            args.submissions_path,
            feature_names,
            threshold=float(best["threshold"]),
            ensemble_logistic_weight=float(best["ensemble_weight"]),
            bias=float(best["bias"]),
            weights=list(best["weights"]),
            means=means,
            stds=stds,
            rules=dict(best["rules"]),
        )
        commit_target = (
            args.submissions_path.parent / "src" / "commit" / "submissions.py"
        )
        if commit_target.exists():
            shutil.copyfile(args.submissions_path, commit_target)

    print(report)
    print(f"saved_report={args.output_path}")
    if args.update_submission:
        print(f"updated_submission={args.submissions_path}")


if __name__ == "__main__":
    main()
