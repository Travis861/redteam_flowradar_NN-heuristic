#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/consolidated_traffic_data.csv-or-directory [prepared_csv] [weights_report]"
  exit 1
fi

INPUT_PATH="$1"
PREPARED_CSV="${2:-./prepared_cic_vpn2016.csv}"
WEIGHTS_REPORT="${3:-./trained_vpn_weights.txt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "[1/2] Preparing CIC-VPN2016 CSV..."
python3 ./scripts/prepare_cic_vpn2016_csv.py \
  --input-path "${INPUT_PATH}" \
  --output-path "${PREPARED_CSV}"

echo "[2/2] Training weights and updating submission..."
python3 ./scripts/train_vpn_weights.py \
  --csv-path "${PREPARED_CSV}" \
  --output-path "${WEIGHTS_REPORT}" \
  --update-submission

echo
echo "Pipeline completed."
echo "Prepared CSV: ${PREPARED_CSV}"
echo "Weights report: ${WEIGHTS_REPORT}"
echo "Updated submission: ./src/commit/submissions.py"
echo
echo "Next:"
echo "  cd ./src"
echo "  uvicorn app:app --reload --host 0.0.0.0 --port 10002"
