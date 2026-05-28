"""Meantone temperament tables and comparison with equal temperament."""

from __future__ import annotations

import numpy as np

from .chord import Chord
from .tension import TensionMeter

# ---------------------------------------------------------------------------
# Standard meantone fifths (in cents) for reference
# ---------------------------------------------------------------------------
_MEANTONE_FIFTHS: dict[str, float] = {
    "quarter-comma": 696.578,   # 4√5 · 700  (quarter-comma meantone)
    "third-comma": 694.786,
    "fifth-comma": 698.371,
    "silbermann": 698.046,
    "werckmeister3": 696.0,    # approximate
    "equal": 700.0,
}

# Pure intervals in cents
_PURE_THIRD = 386.314  # 5/4
_PURE_FIFTH = 701.955  # 3/2


class MeantoneComparison:
    """Compare tension in meantone vs equal temperament."""

    @staticmethod
    def meantone_fifth(ratio: float = 5 / 4) -> float:
        """Compute meantone fifth size (cents) from pure third ratio.

        The meantone fifth is sized so that four fifths up produce a
        pure major third:  4 × fifth = 2 octaves + third.
        """
        third_cents = 1200.0 * np.log2(ratio)
        # 4 fifths = 2 octaves + third  →  fifth = (2400 + third) / 4
        return (2400.0 + third_cents) / 4.0

    @staticmethod
    def compare_tension(chord: Chord, temperament: str = "meantone") -> dict:
        """Compare tension in *temperament* vs equal temperament.

        Returns a dict with ET tension, meantone tension, and difference.
        """
        et_tension = TensionMeter.spectral_tension(chord)

        # For meantone, we approximate by adjusting interval purity
        # Meantone gives purer thirds (5/4) but worse fifths
        # Simple model: reduce tension for chords with major thirds,
        # increase for chords with perfect fifths
        intervals = chord.intervals
        meantone_factor = 1.0
        for iv in intervals:
            if iv == 4:  # major third — purer in meantone
                meantone_factor -= 0.08
            elif iv == 7:  # perfect fifth — slightly worse in meantone
                meantone_factor += 0.02
            elif iv == 3:  # minor third — similar
                meantone_factor -= 0.02
            elif iv == 10:  # minor seventh
                meantone_factor += 0.03

        meantone_tension = et_tension * max(meantone_factor, 0.1)
        return {
            "equal_temperament": et_tension,
            "meantone": meantone_tension,
            "difference": et_tension - meantone_tension,
        }

    @staticmethod
    def wolf_interval(temperament: str = "quarter-comma") -> float:
        """Return the size of the wolf fifth/interval in cents.

        The wolf fifth arises because 12 meantone fifths don't close
        the circle — there's a remainder (the wolf).
        """
        fifth = _MEANTONE_FIFTHS.get(temperament, 696.578)
        # 12 fifths should = 7 octaves (8400 cents)
        # In meantone, 12 × fifth < 8400, the deficit is the wolf
        wolf = 8400.0 - 12.0 * fifth  # wolf is distributed, but one fifth absorbs it
        # The wolf fifth is the normal fifth + the deficit
        return fifth + wolf  # this will be sharp
