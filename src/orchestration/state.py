from typing import Any
from pydantic import BaseModel, Field


class WorkflowState(BaseModel):
    project_id: str = "hotel-review-project"
    current_step: str = "start"
    status: str = "initialized"

    input_paths: list[str] = Field(default_factory=list)
    prior_artifacts: list[str] = Field(default_factory=list)

    approved: bool = False
    warnings: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)

    results: dict[str, Any] = Field(default_factory=dict)