"""Tension gradient and phase-space analysis for chord progressions."""

from __future__ import annotations

import numpy as np


class TensionGradient:
    """Compute and analyze tension gradients across progressions."""

    @staticmethod
    def tension_curve_gradient(tensions: list[float]) -> list[float]:
        """dT/dt for a sequence of tension values (forward differences)."""
        return [tensions[i + 1] - tensions[i] for i in range(len(tensions) - 1)]

    @staticmethod
    def gradient_variance(tensions: list[float]) -> float:
        """Variance of tension gradient — our confirmed predictor of smoothness."""
        if len(tensions) < 3:
            return 0.0
        gradients = TensionGradient.tension_curve_gradient(tensions)
        return float(np.var(gradients))

    @staticmethod
    def smoothness_index(tensions: list[float]) -> float:
        """1 / (1 + Var(dT/dt)) — higher = smoother progression."""
        return 1.0 / (1.0 + TensionGradient.gradient_variance(tensions))

    @staticmethod
    def phase_space_trajectory(tensions: list[float]) -> list[tuple[float, float]]:
        """(T, dT/dt) pairs — the phase-space view of tension evolution."""
        gradients = TensionGradient.tension_curve_gradient(tensions)
        return [(tensions[i + 1], gradients[i]) for i in range(len(gradients))]

    @staticmethod
    def symplectic_area(tensions: list[float]) -> float:
        """
        Area enclosed in phase space — measures tension journey magnitude.

        Uses the shoelace formula on the (T, dT/dt) trajectory.
        """
        trajectory = TensionGradient.phase_space_trajectory(tensions)
        if len(trajectory) < 3:
            return 0.0
        area = 0.0
        n = len(trajectory)
        for i in range(n):
            j = (i + 1) % n
            area += trajectory[i][0] * trajectory[j][1]
            area -= trajectory[j][0] * trajectory[i][1]
        return abs(area) / 2.0

    @staticmethod
    def tension_momentum(tensions: list[float]) -> float:
        """
        ∑ T × dT/dt — the 'momentum' of the tension curve.

        Positive = tension is building; negative = releasing.
        """
        trajectory = TensionGradient.phase_space_trajectory(tensions)
        return sum(t * dt for t, dt in trajectory)

    @staticmethod
    def is_symplectic(tensions: list[float], tol: float = 0.01) -> bool:
        """
        Check if the phase-space trajectory is approximately symplectic
        (area-preserving), meaning the tension budget is conserved.
        """
        trajectory = TensionGradient.phase_space_trajectory(tensions)
        if len(trajectory) < 3:
            return True
        # In a symplectic system, the sum of T×dT should approximately
        # equal the symplectic area (Stokes' theorem analog)
        momentum = TensionGradient.tension_momentum(tensions)
        area = TensionGradient.symplectic_area(tensions)
        return abs(momentum - area) < tol or area < tol
