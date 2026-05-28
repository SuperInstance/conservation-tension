"""Conservation law checking across chord progressions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConservationResult:
    """Result of a conservation-of-tension check."""

    conserved: bool
    total_tension: float
    mean_tension: float
    variance: float
    violation_indices: list[int] = field(default_factory=list)
    max_jump: float = 0.0
    budget_used: float = 0.0


class ConservationLaw:
    """Track conservation of tension across progressions.

    The principle: total harmonic tension tends to be approximately conserved
    (or smoothly varying) across well-formed chord progressions.  Sudden
    jumps indicate a violation — a moment where the musical grammar breaks.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def check(self, tensions: list[float]) -> ConservationResult:
        """Check if tension is approximately conserved."""
        if not tensions:
            return ConservationResult(
                conserved=True, total_tension=0.0,
                mean_tension=0.0, variance=0.0,
            )
        viols = self.violations(tensions)
        mean_t = sum(tensions) / len(tensions)
        var_t = sum((t - mean_t) ** 2 for t in tensions) / len(tensions)
        total_t = sum(tensions)
        budget = self.tension_budget(tensions)
        max_jump = 0.0
        for i in range(1, len(tensions)):
            jump = abs(tensions[i] - tensions[i - 1])
            if jump > max_jump:
                max_jump = jump
        return ConservationResult(
            conserved=len(viols) == 0,
            total_tension=total_t,
            mean_tension=mean_t,
            variance=var_t,
            violation_indices=viols,
            max_jump=max_jump,
            budget_used=budget.get("budget_used", 0.0),
        )

    def violations(self, tensions: list[float]) -> list[int]:
        """Return indices *i* where |tension[i] - tension[i-1]| > threshold."""
        viols: list[int] = []
        for i in range(1, len(tensions)):
            if abs(tensions[i] - tensions[i - 1]) > self.threshold:
                viols.append(i)
        return viols

    def tension_budget(self, tensions: list[float]) -> dict:
        """Compute tension budget statistics."""
        if not tensions:
            return {"total": 0.0, "average": 0.0, "variance": 0.0, "budget_used": 0.0}
        total = sum(tensions)
        avg = total / len(tensions)
        var = sum((t - avg) ** 2 for t in tensions) / len(tensions)
        # Budget used: ratio of actual variance to maximum possible variance
        max_var = (max(tensions) - min(tensions)) ** 2 / 4 if len(tensions) > 1 else 1.0
        budget_used = var / max_var if max_var > 0 else 0.0
        return {
            "total": total,
            "average": avg,
            "variance": var,
            "budget_used": budget_used,
        }
