from typing import Any

from pydantic import BaseModel, Field, field_validator


class MinerInput(BaseModel):
    random_val: str | None = Field(
        default=None,
        min_length=4,
        max_length=64,
        title="Random Value",
        description="Random value to prevent caching.",
        examples=["a1b2c3d4e5f6g7h8"],
    )


class MinerOutput(BaseModel):
    commit_files: str = Field(
        ...,
        title="Commit Files",
        description="List of Commit files for the challenge.",
    )

    @field_validator("commit_files", mode="after")
    @classmethod
    def _check_commit_files(cls, val: str) -> str:
        _content_lines = val.splitlines()
        if len(_content_lines) > 1000:
            raise ValueError(
                f"Commit files contain too many lines, should be <= 1000 lines!"
            )

        return val


class FingerprintRequest(BaseModel):
    products: dict[str, Any] = Field(
        ...,
        title="Flow Features",
        description="Network flow features used for VPN detection.",
        examples=[
            {
                "flow_duration": 1504,
                "fwd_num_pkts": 11,
                "bwd_num_pkts": 10,
                "fwd_sum_pkt_len": 3211,
                "bwd_sum_pkt_len": 1334,
            }
        ],
    )


class FingerprintResponse(BaseModel):
    is_vpn: bool = Field(..., title="Is VPN", description="Predicted VPN label.")
    request_id: str = Field(..., title="Request ID", description="Unique request identifier.")
    vpn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        title="VPN Probability",
        description="Probability-like VPN score from the detector.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        title="Reasons",
        description="Human-readable explanation tags for the prediction.",
    )
    engineered_features: dict[str, float] = Field(
        default_factory=dict,
        title="Engineered Features",
        description="Derived features used by the local detector.",
    )


__all__ = [
    "FingerprintRequest",
    "FingerprintResponse",
    "MinerInput",
    "MinerOutput",
]
