# Miner Commit - FlowRadar: VPN Detection

This is a miner commit API example for FlowRadar: VPN Detection.

## ✨ Features

- Miner commit
- Health check endpoint
- FastAPI
- Web service

---

## 🛠 Installation

### 1. 🚧 Prerequisites

- Install **Python (>= v3.10)** and **pip (>= 23)**:
    - **[RECOMMENDED] [Miniconda (v3)](https://www.anaconda.com/docs/getting-started/miniconda/install)**
    - *[arm64/aarch64] [Miniforge (v3)](https://github.com/conda-forge/miniforge)*
    - *[Python virtual environment] [venv](https://docs.python.org/3/library/venv.html)*

[OPTIONAL] For **DEVELOPMENT** environment:

- Install [**git**](https://git-scm.com/downloads)
- Setup an [**SSH key**](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)

### 2. 📦 Install dependencies

```sh
pip install -r ./requirements.txt
```

### 3. 🏁 Start the server

```sh
cd src
uvicorn app:app --host="0.0.0.0" --port=10002 --no-access-log --no-server-header --proxy-headers --forwarded-allow-ips="*"

# For DEVELOPMENT:
uvicorn app:app --host="0.0.0.0" --port=10002 --no-access-log --no-server-header --proxy-headers --forwarded-allow-ips="*" --reload
```

### 4. ✅ Check server is running

Check with CLI (curl):

```sh
# Send a ping request with 'curl' to API server:
curl -s http://localhost:10002/ping
```

Check with web browser:

- Health check: <http://localhost:10002/health>
- Swagger: <http://localhost:10002/docs>
- Redoc: <http://localhost:10002/redoc>
- OpenAPI JSON: <http://localhost:10002/openapi.json>

Test the local VPN endpoint:

```sh
curl -X POST http://localhost:10002/fingerprint \
  -H "Content-Type: application/json" \
  -d '{
    "products": {
      "flow_duration": 1504,
      "fwd_num_pkts": 11,
      "bwd_num_pkts": 10,
      "fwd_sum_pkt_len": 3211,
      "bwd_sum_pkt_len": 1334
    }
  }'
```

### 5. Train Better Detector Weights

The example detector in `src/commit/submissions.py` includes a lightweight
linear scoring layer with hand-tuned defaults. You can fit better weights from
your own labeled CSV and paste the exported block back into the submission.

If you downloaded the CIC-VPN2016 Kaggle CSV, first normalize it into the small
trainer-friendly format expected by the example:

```sh
python ./scripts/prepare_cic_vpn2016_csv.py \
  --input-path /path/to/cic-vpn2016.csv \
  --output-path ./prepared_cic_vpn2016.csv
```

You can also point `--input-path` at a directory and the script will combine all
CSV files it finds there.

Shortest path from Kaggle CSV to updated submission:

```sh
bash ./scripts/run_full_local_pipeline.sh /path/to/cic-vpn2016.csv
```

Expected CSV shape:

- one row per flow
- label column such as `is_vpn`, `label`, `vpn`, `target`, or `class`
- raw flow fields such as `flow_duration`, `fwd_num_pkts`, `bwd_num_pkts`,
  `fwd_sum_pkt_len`, `bwd_sum_pkt_len`

Train weights:

```sh
python ./scripts/train_vpn_weights.py \
  --csv-path /path/to/vpn_flows.csv \
  --output-path ./trained_vpn_weights.txt
```

Train and update the submission automatically:

```sh
python ./scripts/train_vpn_weights.py \
  --csv-path ./prepared_cic_vpn2016.csv \
  --output-path ./trained_vpn_weights.txt \
  --update-submission
```

The script:

- engineers the same features used by the submission
- fits a lightweight logistic model in pure Python
- prints a `MODEL_WEIGHTS = {...}` block
- saves the same report to `trained_vpn_weights.txt`
- can update `src/commit/submissions.py` automatically with `--update-submission`

After training, replace the `MODEL_WEIGHTS` block in
`src/commit/submissions.py` with the exported one, or let the script do it
for you with `--update-submission`.

---

## 🏗️ Build Docker Image

To build the docker image, run the following command:

```sh
docker build -t myhub/rest-flr-commit:0.0.1 .

# For MacOS (Apple Silicon) to build AMD64:
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t myhub/rest-flr-commit:0.0.1 .
```
