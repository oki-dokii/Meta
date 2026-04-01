"""
models.py
=========
Typed Pydantic data-models for the ContentModerationEnv OpenEnv spec.

These models are used for:
  • runtime validation of agent actions
  • serialisation of environment state
  • documentation (JSON Schema derivable via .model_json_schema())

Requires: pydantic >= 2.0
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class Label(str, Enum):
    """Content classification label."""
    safe       = "safe"
    toxic      = "toxic"
    spam       = "spam"
    misleading = "misleading"


class ModerationAction(str, Enum):
    """Moderation action to apply to the content."""
    allow      = "allow"
    warn       = "warn"
    remove     = "remove"
    shadowban  = "shadowban"
    escalate   = "escalate"


class PlatformPolicy(str, Enum):
    """Enforcement level of the platform."""
    strict   = "strict"
    moderate = "moderate"
    lenient  = "lenient"


class Tier(str, Enum):
    """Benchmark difficulty tier."""
    easy   = "easy"
    medium = "medium"
    hard   = "hard"


# ── Observation (state) ───────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    The environment's observation returned by reset() / state().

    Attributes
    ----------
    text : str
        The user-generated text content to be reviewed.
    audio_transcript : str | None
        Transcript of any accompanying audio/video (None for text-only).
    visual_tags : list[str]
        Machine-detected visual content tags (empty list if no visual media).
    previous_flags : int
        Number of prior platform violations by this account (≥ 0).
    platform_policy : PlatformPolicy
        Policy enforcement level the moderation decision must respect.
    """
    text:              str              = Field(..., description="User-generated content text")
    audio_transcript:  Optional[str]    = Field(None, description="Audio/video transcript (nullable)")
    visual_tags:       List[str]        = Field(default_factory=list, description="Detected visual content tags")
    previous_flags:    int              = Field(..., ge=0, description="Prior policy violations count")
    platform_policy:   PlatformPolicy  = Field(..., description="Platform enforcement level")

    model_config = {"frozen": True}   # immutable — agents must not mutate state


# ── Action ────────────────────────────────────────────────────────────────────

class AgentAction(BaseModel):
    """
    The action an agent submits via env.step().

    Required
    --------
    label    : Label
    action   : ModerationAction

    Optional (scored only in hard tier)
    ------------------------------------
    severity  : int in [1, 5]
    rationale : str
    """
    label:     Label            = Field(..., description="Content classification")
    action:    ModerationAction = Field(..., description="Moderation action to apply")
    severity:  Optional[int]    = Field(None, ge=1, le=5, description="Severity 1-5 (hard tier only)")
    rationale: Optional[str]    = Field(None, description="Brief reasoning (not scored)")

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v):
        """Accept string integers gracefully."""
        if v is not None:
            return int(v)
        return v

    def to_env_dict(self) -> dict:
        """Convert to the plain dict format expected by ContentModerationEnv.step()."""
        d: dict = {"label": self.label.value, "action": self.action.value}
        if self.severity is not None:
            d["severity"] = self.severity
        if self.rationale is not None:
            d["rationale"] = self.rationale
        return d


# ── Score breakdown ───────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    """Per-component reward breakdown returned in step() info dict."""
    label_correct:    Optional[float] = Field(None, ge=0.0, le=1.0)
    action_correct:   Optional[float] = Field(None, ge=0.0, le=1.0)
    severity_within_1: Optional[float] = Field(None, ge=0.0, le=1.0)

    @property
    def total(self) -> float:
        return sum(
            v for v in [self.label_correct, self.action_correct, self.severity_within_1]
            if v is not None
        )


# ── Step result ───────────────────────────────────────────────────────────────

class GroundTruth(BaseModel):
    """Ground truth record stored in each scenario."""
    label:     Label
    action:    ModerationAction
    severity:  Optional[int]  = Field(None, ge=1, le=5)
    rationale: Optional[str]  = None


class StepResult(BaseModel):
    """
    Full result returned by env.step().

    state   — the original observation (unchanged after step)
    reward  — 0.0 … 1.0 partial-credit score
    done    — always True (single-step episodes)
    info    — breakdown, ground truth, submitted action, warnings
    """
    state:   Observation
    reward:  float = Field(..., ge=0.0, le=1.0)
    done:    bool  = True
    info:    StepInfo

    @model_validator(mode="after")
    def done_is_always_true(self):
        assert self.done is True, "done must always be True in single-step episodes"
        return self


class StepInfo(BaseModel):
    """Metadata returned inside StepResult.info."""
    scenario_id:      str
    tier:             Tier
    ground_truth:     GroundTruth
    score_rubric:     dict
    score_breakdown:  ScoreBreakdown
    submitted_action: AgentAction
    warnings:         List[str] = Field(default_factory=list)


# ── Scenario (internal) ───────────────────────────────────────────────────────

class Scenario(BaseModel):
    """
    One benchmark scenario as stored in moderation_benchmark.json.
    Used internally by the environment loader.
    """
    id:           str
    tier:         Tier
    state:        Observation
    ground_truth: GroundTruth
    score_rubric: dict
