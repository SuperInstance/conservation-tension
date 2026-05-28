"""Eigenbasis rotation for tension measurements."""

from __future__ import annotations

import numpy as np

from .chord import Chord
from .tension import TensionMeter
from .progression import ChordProgression


class TensionEigenbasis:
    """
    Rotate tension measurements into the eigenbasis where conservation holds.

    The three tension components (spectral, voice-leading, contextual)
    are correlated. By rotating into the principal-component basis,
    we find the direction where tension conservation is strongest —
    the Eigenbasis Hypothesis.
    """

    @staticmethod
    def tension_tensor(progressions: list[ChordProgression],
                       key: str = "C") -> np.ndarray:
        """
        Build a 3×N tensor of (spectral, voice-leading, contextual) tensions
        across all chords in all progressions.

        Returns shape (3, N) where N = total number of chords.
        """
        all_spectral = []
        all_voice_leading = []
        all_contextual = []

        for prog in progressions:
            chords = prog.chords
            for i, chord in enumerate(chords):
                all_spectral.append(TensionMeter.spectral_tension(chord))

                if i > 0:
                    all_voice_leading.append(
                        TensionMeter.voice_leading_tension(chords[i - 1], chord)
                    )
                else:
                    all_voice_leading.append(0.0)

                all_contextual.append(
                    TensionMeter.contextual_tension(chord, key)
                )

        return np.array([all_spectral, all_voice_leading, all_contextual])

    @staticmethod
    def eigenbasis_rotation(progressions: list[ChordProgression],
                            key: str = "C") -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the rotation matrix from the correlation structure.

        Returns (eigenvalues, eigenvectors) where eigenvectors are columns
        of the rotation matrix, ordered by eigenvalue (ascending).
        """
        tensor = TensionEigenbasis.tension_tensor(progressions, key)
        if tensor.shape[1] < 3:
            # Not enough data for meaningful correlation
            return np.array([1.0, 1.0, 1.0]), np.eye(3)

        corr = np.corrcoef(tensor)

        # Handle any NaN from constant rows
        corr = np.nan_to_num(corr, nan=0.0)
        # Ensure positive semi-definite
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)

        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        return eigenvalues, eigenvectors

    @staticmethod
    def projected_tension(tensions: dict[str, float],
                          rotation: np.ndarray) -> float:
        """
        Project a tension vector into the principal eigenbasis direction.

        tensions must have keys: 'spectral', 'voice_leading', 'contextual'.
        rotation is the eigenvector matrix from eigenbasis_rotation().
        """
        vector = np.array([
            tensions.get('spectral', 0.0),
            tensions.get('voice_leading', 0.0),
            tensions.get('contextual', 0.0),
        ])
        projected = rotation @ vector
        # Return the component along the principal (largest eigenvalue) direction
        # eigh returns ascending order, so last column is principal
        return float(projected[-1])

    @staticmethod
    def explained_variance(eigenvalues: np.ndarray) -> np.ndarray:
        """
        Fraction of total variance explained by each eigenvector.
        """
        total = np.sum(eigenvalues)
        if total == 0:
            return np.zeros_like(eigenvalues)
        return eigenvalues / total
