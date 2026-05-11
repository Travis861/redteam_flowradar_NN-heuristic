import sys
import uuid
import logging
import pathlib

from fastapi import FastAPI, Body, HTTPException
from data_types import (
    FingerprintRequest,
    FingerprintResponse,
    MinerInput,
    MinerOutput,
)
from commit.submissions import detect_vpn_details

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S %z",
    format="[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d]: %(message)s",
)


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fingerprint", response_model=FingerprintResponse)
def fingerprint(payload: FingerprintRequest = Body(...)) -> FingerprintResponse:
    request_id = str(uuid.uuid4())
    logger.info("Processing fingerprint request | request_id=%s", request_id)

    try:
        result = detect_vpn_details(payload.products)
        response = FingerprintResponse(
            is_vpn=bool(result["is_vpn"]),
            request_id=request_id,
            vpn_probability=float(result["vpn_probability"]),
            reasons=list(result.get("reasons", [])),
            engineered_features=dict(result.get("engineered_features", {})),
        )
        logger.info(
            "Fingerprint request completed | request_id=%s is_vpn=%s vpn_probability=%.3f",
            request_id,
            response.is_vpn,
            response.vpn_probability,
        )
        return response
    except Exception as err:
        logger.exception("Fingerprint request failed | request_id=%s error=%s", request_id, err)
        raise HTTPException(status_code=500, detail="Failed to process fingerprint request.")


@app.post("/solve", response_model=MinerOutput)
def solve(miner_input: MinerInput = Body(...)) -> MinerOutput:

    logger.info("Retrieving commit files...")
    _miner_output: MinerOutput
    try:
        _src_dir = pathlib.Path(__file__).parent.resolve()
        _commit_dir = _src_dir / "commit" / "submissions.py"
        _commit_file_pm = ""
        with open(_commit_dir) as _commit_file:
            _commit_file_pm = _commit_file.read()

        _miner_output = MinerOutput(commit_files=_commit_file_pm)
        logger.info("Successfully retrieved commit files.")
    except Exception as err:
        logger.error(f"Failed to retrieve commit files: {str(err)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve commit files.")

    return _miner_output


__all__ = ["app"]
