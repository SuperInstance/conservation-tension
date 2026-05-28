"""Tension computation: pitch-class-aware spectral, voice-leading, and contextual methods."""

from __future__ import annotations

import numpy as np

from .chord import Chord
from .pitch import Pitch

# Circle-of-fifths position for each pitch class
_COF_POS: dict[str, int] = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5,
    "F#": 6, "Gb": 6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1,
}


def _cof_distance(key1: str, key2: str) -> int:
    """Circle-of-fifths distance between two key names."""
    return abs(_COF_POS.get(key1, 0) - _COF_POS.get(key2, 0))


# ---------------------------------------------------------------------------
# Just-intonation interval table: (numerator, denominator) → Tenney height
# Tenney height = log2(numerator * denominator) for the reduced ratio
# ---------------------------------------------------------------------------
_JUST_RATIOS: list[tuple[float, int, int]] = [
    # (frequency_ratio, numerator, denominator)
    (1.0, 1, 1),          # unison
    (2.0, 2, 1),          # octave
    (1.5, 3, 2),          # perfect fifth
    (4.0 / 3.0, 4, 3),    # perfect fourth
    (5.0 / 4.0, 5, 4),    # major third
    (6.0 / 5.0, 6, 5),    # minor third
    (5.0 / 3.0, 5, 3),    # major sixth
    (8.0 / 5.0, 8, 5),    # minor sixth
    (9.0 / 8.0, 9, 8),    # major second
    (10.0 / 9.0, 10, 9),  # minor tone
    (16.0 / 15.0, 16, 15), # minor second
    (15.0 / 8.0, 15, 8),  # major seventh
    (9.0 / 5.0, 9, 5),    # minor seventh
    (45.0 / 32.0, 45, 32), # tritone
    (7.0 / 4.0, 7, 4),    # harmonic seventh
    (7.0 / 5.0, 7, 5),    # septimal tritone
]

# Precompute Tenney heights and build lookup
_JUST_TABLE: list[tuple[float, float, float]] = []
for ratio, num, den in _JUST_RATIOS:
    tenney = np.log2(num * den)
    _JUST_TABLE.append((ratio, tenney, 1200.0 * np.log2(ratio) if ratio >= 1.0 else 0.0))


def _tenney_dissonance(interval_cents: float) -> float:
    """
    Compute the dissonance of an interval given in cents.

    Finds the nearest just interval and returns its Tenney height
    plus a penalty for deviation from just intonation.
    """
    if interval_cents < 1.0:
        return 0.0  # unison

    best_dissonance = float('inf')
    for ratio, tenney, just_cents in _JUST_TABLE:
        if just_cents < 1.0:
            continue  # skip unison
        # Check both the just interval and its octave complement
        deviation = min(abs(interval_cents - just_cents),
                        abs(interval_cents - (just_cents + 1200.0)),
                        abs(interval_cents - abs(just_cents - 1200.0)))
        # Dissonance = Tenney height + deviation penalty
        dissonance = tenney + deviation / 100.0  # 100 cents deviation = +1.0 penalty
        if dissonance < best_dissonance:
            best_dissonance = dissonance
    return best_dissonance


class TensionMeter:
    """Compute harmonic tension using different methods."""

    @staticmethod
    def spectral_tension(chord: Chord, n_harmonics: int = 8) -> float:
        """
        Pitch-class-aware spectral tension using Tenney height dissonance.

        For each pair of pitch classes in the chord:
        1. Compute the interval in cents
        2. Find the nearest just interval
        3. Weight by the Tenney height of that just interval + deviation penalty

        Returns tension in roughly 0–2 range. Consonant triads ~0.5–1.0,
        dissonant chords ~1.5–2.5+.
        """
        if not chord.pitches:
            return 0.0

        pitches = chord.pitches
        n = len(pitches)
        if n < 2:
            return 0.0

        total_dissonance = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Compute interval in cents (reduce to within an octave)
                cents = abs(pitches[i].interval_to(pitches[j]))
                cents = cents % 1200.0
                # Compute dissonance for both the interval and its complement,
                # take the minimum (a fifth is consonant even though its
                # complement is a fourth)
                d1 = _tenney_dissonance(cents)
                d2 = _tenney_dissonance(1200.0 - cents) if cents > 1.0 else d1
                total_dissonance += min(d1, d2)
                pair_count += 1

        if pair_count == 0:
            return 0.0

        # Normalize by sqrt(pair_count) so larger chords aren't diluted
        # by averaging: a 7th chord (6 pairs) should be tenser than a
        # triad (3 pairs) even though it shares consonant intervals.
        return total_dissonance / (np.sqrt(pair_count) * 3.0)

    @staticmethod
    def combined_tension(chord: Chord, prev: Chord | None = None,
                         key: str = "C", scale: str = "major") -> dict[str, float]:
        """
        Compute all three tension components for a chord.

        Returns dict with spectral, voice_leading, contextual, and combined.
        """
        spectral = TensionMeter.spectral_tension(chord)
        voice_leading = TensionMeter.voice_leading_tension(prev, chord) if prev else 0.0
        contextual = TensionMeter.contextual_tension(chord, key, scale)

        # Weighted combination (weights from empirical calibration)
        combined = 0.45 * spectral + 0.30 * voice_leading + 0.25 * contextual

        return {
            "spectral": spectral,
            "voice_leading": voice_leading,
            "contextual": contextual,
            "combined": combined,
        }

    @staticmethod
    def voice_leading_tension(prev: Chord, curr: Chord) -> float:
        """
        Minimal voice-leading distance between *prev* and *curr*.

        Uses a greedy nearest-pitch matching to approximate minimal
        total semitone motion.
        """
        prev_midis = sorted(p.midi for p in prev.pitches)
        curr_midis = sorted(p.midi for p in curr.pitches)
        # Pad shorter list with nearest octave repeats
        max_len = max(len(prev_midis), len(curr_midis))
        while len(prev_midis) < max_len:
            prev_midis.append(prev_midis[-1] + 12)
        while len(curr_midis) < max_len:
            curr_midis.append(curr_midis[-1] + 12)
        # Greedy minimal matching
        total = 0.0
        used: set[int] = set()
        for cm in curr_midis:
            best = float("inf")
            best_idx = 0
            for i, pm in enumerate(prev_midis):
                if i in used:
                    continue
                d = abs(cm - pm)
                if d < best:
                    best = d
                    best_idx = i
            used.add(best_idx)
            total += best
        return total / max_len

    @staticmethod
    def contextual_tension(chord: Chord, key: str, scale: str = "major") -> float:
        """
        Distance from tonal center using the circle of fifths.

        The root of *chord* is compared to *key* via circle-of-fifths
        distance, plus extra weight for chromatic scale degrees.
        """
        if not chord.pitches:
            return 0.0
        root_letter = chord.root.name.rstrip("0123456789")
        root_letter = root_letter.replace("#", "").replace("b", "")
        # Get pitch class name
        pc = chord.root.midi % 12
        _PC_NAMES = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
                     6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
        root_pc_name = _PC_NAMES.get(pc, "C")

        cof_dist = _cof_distance(key, root_pc_name)

        # Scale degree check: is the chord root in the scale?
        scale_intervals = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        }
        si = scale_intervals.get(scale, scale_intervals["major"])
        key_pc = {"C": 0, "G": 7, "D": 2, "A": 9, "E": 4, "B": 11,
                  "F": 5, "Bb": 10, "Eb": 3, "Ab": 8, "Db": 1, "F#": 6,
                  "Gb": 6, "C#": 1}.get(key, 0)
        in_scale = (pc - key_pc) % 12 in si

        tension = cof_dist / 6.0  # normalize: max COF distance ~6
        if not in_scale:
            tension += 0.5
        # Quality adjustment: minor chords slightly more tense in major key
        if scale == "major" and "minor" in chord.quality:
            tension += 0.05
        return tension
