#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_OUTPUT_COLUMNS = [
    "flow_duration",
    "fwd_num_pkts",
    "bwd_num_pkts",
    "fwd_sum_pkt_len",
    "bwd_sum_pkt_len",
    "is_vpn",
    "source_label",
]

COLUMN_ALIASES = {
    "flow_duration": [
        "flow_duration",
        "flow duration",
        "duration",
    ],
    "fwd_num_pkts": [
        "fwd_num_pkts",
        "total fwd packets",
        "tot fwd pkts",
        "forward packet count",
        "total forward packets",
    ],
    "bwd_num_pkts": [
        "bwd_num_pkts",
        "total backward packets",
        "tot bwd pkts",
        "backward packet count",
        "total bwd packets",
    ],
    "fwd_sum_pkt_len": [
        "fwd_sum_pkt_len",
        "total length of fwd packets",
        "totlen fwd pkts",
        "fwd packet length total",
    ],
    "bwd_sum_pkt_len": [
        "bwd_sum_pkt_len",
        "total length of bwd packets",
        "totlen bwd pkts",
        "bwd packet length total",
    ],
    "label": [
        "traffic_type",
        "label",
        "class",
        "category",
    ],
    "flow_pkts_per_second": [
        "flowpktspersecond",
        "flow pkts per second",
        "flow packets per second",
    ],
    "flow_bytes_per_second": [
        "flowbytespersecond",
        "flow bytes per second",
    ],
    "total_fiat": [
        "total_fiat",
        "total fiat",
    ],
    "total_biat": [
        "total_biat",
        "total biat",
    ],
    "mean_fiat": [
        "mean_fiat",
        "mean fiat",
    ],
    "mean_biat": [
        "mean_biat",
        "mean biat",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CIC-VPN2016 Kaggle CSV files for the FlowRadar weight trainer."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="A CIC-VPN2016 CSV file or a directory containing CSV files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("prepared_cic_vpn2016.csv"),
        help="Normalized output CSV path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a row cannot be normalized. By default, invalid rows are skipped.",
    )
    return parser.parse_args()


def _normalize_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.csv") if path.is_file())
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _find_column(fieldnames: list[str], canonical_name: str, *, required: bool = True) -> str | None:
    normalized_map = {_normalize_name(name): name for name in fieldnames}
    for alias in COLUMN_ALIASES[canonical_name]:
        matched = normalized_map.get(_normalize_name(alias))
        if matched:
            return matched
    if required:
        raise KeyError(
            f"Could not find a column for '{canonical_name}'. Available columns: {', '.join(fieldnames)}"
        )
    return None


def _label_to_binary(label: object) -> tuple[int, str]:
    text = str(label or "").strip()
    normalized = text.lower()
    if not normalized:
        return 0, text
    if normalized.startswith("vpn"):
        return 1, text
    if "vpn-" in normalized or normalized == "vpn":
        return 1, text
    if normalized.startswith("nonvpn") or normalized.startswith("non-vpn"):
        return 0, text
    if "nonvpn" in normalized or "non-vpn" in normalized:
        return 0, text
    # For binary numeric labels.
    if normalized in {"1", "true", "yes"}:
        return 1, text
    if normalized in {"0", "false", "no"}:
        return 0, text
    # If we cannot infer from the label text, default to non-VPN to avoid false positives.
    return 0, text


def _normalize_row(row: dict[str, str], column_map: dict[str, str]) -> dict[str, object]:
    is_vpn, source_label = _label_to_binary(row.get(column_map["label"]))
    flow_duration = _safe_float(row.get(column_map["flow_duration"]))
    duration_s = flow_duration / 1_000_000.0 if flow_duration > 10_000 else flow_duration / 1000.0

    fwd_num_pkts = _safe_float(row.get(column_map["fwd_num_pkts"])) if column_map.get("fwd_num_pkts") else 0.0
    bwd_num_pkts = _safe_float(row.get(column_map["bwd_num_pkts"])) if column_map.get("bwd_num_pkts") else 0.0
    fwd_sum_pkt_len = _safe_float(row.get(column_map["fwd_sum_pkt_len"])) if column_map.get("fwd_sum_pkt_len") else 0.0
    bwd_sum_pkt_len = _safe_float(row.get(column_map["bwd_sum_pkt_len"])) if column_map.get("bwd_sum_pkt_len") else 0.0

    # Fallback for aggregate-only schemas like consolidated_traffic_data.csv.
    if not (fwd_num_pkts and bwd_num_pkts):
        flow_pkts_per_second = _safe_float(row.get(column_map["flow_pkts_per_second"])) if column_map.get("flow_pkts_per_second") else 0.0
        total_pkts = flow_pkts_per_second * max(duration_s, 0.0)

        total_fiat = _safe_float(row.get(column_map["total_fiat"])) if column_map.get("total_fiat") else 0.0
        total_biat = _safe_float(row.get(column_map["total_biat"])) if column_map.get("total_biat") else 0.0
        mean_fiat = _safe_float(row.get(column_map["mean_fiat"])) if column_map.get("mean_fiat") else 0.0
        mean_biat = _safe_float(row.get(column_map["mean_biat"])) if column_map.get("mean_biat") else 0.0

        est_fwd = total_fiat / mean_fiat if total_fiat > 0 and mean_fiat > 0 else 0.0
        est_bwd = total_biat / mean_biat if total_biat > 0 and mean_biat > 0 else 0.0
        est_total = est_fwd + est_bwd

        if est_total > 0 and total_pkts > 0:
            fwd_share = est_fwd / est_total
        else:
            fwd_share = 0.5
        fwd_num_pkts = total_pkts * fwd_share
        bwd_num_pkts = max(total_pkts - fwd_num_pkts, 0.0)

    if not (fwd_sum_pkt_len and bwd_sum_pkt_len):
        flow_bytes_per_second = _safe_float(row.get(column_map["flow_bytes_per_second"])) if column_map.get("flow_bytes_per_second") else 0.0
        total_bytes = flow_bytes_per_second * max(duration_s, 0.0)
        total_pkts = max(fwd_num_pkts + bwd_num_pkts, 0.0)
        fwd_share = (fwd_num_pkts / total_pkts) if total_pkts > 0 else 0.5
        fwd_sum_pkt_len = total_bytes * fwd_share
        bwd_sum_pkt_len = max(total_bytes - fwd_sum_pkt_len, 0.0)

    return {
        "flow_duration": flow_duration,
        "fwd_num_pkts": fwd_num_pkts,
        "bwd_num_pkts": bwd_num_pkts,
        "fwd_sum_pkt_len": fwd_sum_pkt_len,
        "bwd_sum_pkt_len": bwd_sum_pkt_len,
        "is_vpn": is_vpn,
        "source_label": source_label,
    }


def prepare_files(input_files: list[Path], output_path: Path, *, strict: bool) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    skipped_rows = 0
    file_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_OUTPUT_COLUMNS)
        writer.writeheader()

        for csv_path in input_files:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
                reader = csv.DictReader(source_handle)
                fieldnames = list(reader.fieldnames or [])
                if not fieldnames:
                    if strict:
                        raise RuntimeError(f"CSV has no header: {csv_path}")
                    skipped_rows += 1
                    continue

                required_columns = {
                    "flow_duration",
                    "label",
                }
                column_map = {
                    canonical: _find_column(
                        fieldnames,
                        canonical,
                        required=canonical in required_columns,
                    )
                    for canonical in COLUMN_ALIASES
                }

                file_count += 1
                for raw_row in reader:
                    try:
                        normalized = _normalize_row(raw_row, column_map)
                        writer.writerow(normalized)
                        written_rows += 1
                    except Exception:
                        if strict:
                            raise
                        skipped_rows += 1

    return {
        "files": file_count,
        "written_rows": written_rows,
        "skipped_rows": skipped_rows,
    }


def main() -> None:
    args = parse_args()
    input_files = _resolve_input_files(args.input_path)
    if not input_files:
        raise RuntimeError(f"No CSV files found under: {args.input_path}")

    stats = prepare_files(input_files, args.output_path, strict=args.strict)
    print(f"input_files={stats['files']}")
    print(f"written_rows={stats['written_rows']}")
    print(f"skipped_rows={stats['skipped_rows']}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
