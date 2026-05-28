"""ChordProgression with tension tracking and full analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chord import Chord
from .tension import TensionMeter
from .conservation import ConservationLaw, ConservationResult

import numpy as np


@dataclass
class ProgressionAnalysis:
    """Full analysis result of a chord progression."""

    chords: list[str]
    tension_curve: list[float]
    cumulative_tension: list[float]
    conservation_result: ConservationResult
    conservation_score: float
    cadences: list[dict] = field(default_factory=list)
    key_areas: list[dict] = field(default_factory=list)


class ChordProgression:
    """A sequence of chords with tension tracking."""

    def __init__(self, chords: list[str]):
        # Accept "Cmaj7 | Dm7 | G7" style or list form
        if len(chords) == 1 and "|" in chords[0]:
            chords = [c.strip() for c in chords[0].split("|")]
        self._symbols = chords
        self._chords = [Chord.parse(s) for s in chords]

    @property
    def chords(self) -> list[Chord]:
        return list(self._chords)

    def analyze(self, key: str = "C") -> ProgressionAnalysis:
        """Full analysis: tension curve, conservation, key changes."""
        tc = self.tension_curve()
        cum = self.cumulative_tension()
        law = ConservationLaw(threshold=0.1)
        result = law.check(tc)
        score = self.conservation_score()
        cads = self.cadences()
        keys = self.key_areas(key)
        return ProgressionAnalysis(
            chords=self._symbols,
            tension_curve=tc,
            cumulative_tension=cum,
            conservation_result=result,
            conservation_score=score,
            cadences=cads,
            key_areas=keys,
        )

    def tension_curve(self) -> list[float]:
        """Return per-chord spectral tension values."""
        return [TensionMeter.spectral_tension(c) for c in self._chords]

    def cumulative_tension(self) -> list[float]:
        """Return running total of tension."""
        tc = self.tension_curve()
        cum = []
        total = 0.0
        for t in tc:
            total += t
            cum.append(total)
        return cum

    def conservation_score(self) -> float:
        """Return 0–1 score: how well tension is conserved (smooth).

        1.0 = perfectly smooth, 0.0 = maximally erratic.
        """
        tc = self.tension_curve()
        if len(tc) < 2:
            return 1.0
        diffs = [abs(tc[i] - tc[i - 1]) for i in range(1, len(tc))]
        mean_diff = sum(diffs) / len(diffs)
        # Normalize: if mean diff is 0 → score 1; if mean diff is large → score 0
        score = max(0.0, 1.0 - mean_diff / (sum(tc) / len(tc) + 1e-9))
        return min(score, 1.0)

    def cadences(self) -> list[dict]:
        """Detect common cadence patterns (V-I, ii-V-I, etc.)."""
        symbols = self._symbols
        cads: list[dict] = []
        for i in range(len(symbols)):
            # Look for ii-V-I (3-chord window)
            if i + 2 < len(symbols):
                a, b, c = symbols[i], symbols[i + 1], symbols[i + 2]
                if self._is_ii_v_i(a, b, c):
                    cads.append({"type": "ii-V-I", "index": i})
            # V-I (2-chord)
            if i + 1 < len(symbols):
                a, b = symbols[i], symbols[i + 1]
                if self._is_v_i(a, b):
                    cads.append({"type": "V-I", "index": i})
        return cads

    def key_areas(self, default_key: str = "C") -> list[dict]:
        """Estimate key areas by finding tension minima."""
        tc = self.tension_curve()
        if not tc:
            return [{"key": default_key, "start": 0, "end": 0}]
        # Naive: use contextual tension to detect key shifts
        areas: list[dict] = []
        current_key = default_key
        start = 0
        for i, chord in enumerate(self._chords):
            ctx = TensionMeter.contextual_tension(chord, current_key)
            if ctx > 0.6 and i > 0:
                areas.append({"key": current_key, "start": start, "end": i - 1})
                # Guess new key from chord root
                _PC_NAMES = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
                             6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
                pc = chord.root.midi % 12
                current_key = _PC_NAMES.get(pc, current_key)
                start = i
        areas.append({"key": current_key, "start": start, "end": len(tc) - 1})
        return areas

    # ------------------------------------------------------------------
    # Cadence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _root_pc(symbol: str) -> int:
        """Extract root pitch class from a chord symbol."""
        root = symbol.strip()[0].upper()
        _ROOT_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        pc = _ROOT_PC.get(root, 0)
        if len(symbol) > 1 and symbol[1] == "#":
            pc = (pc + 1) % 12
        elif len(symbol) > 1 and symbol[1] == "b":
            pc = (pc - 1) % 12
        return pc

    def _is_v_i(self, a: str, b: str) -> bool:
        pc_a = self._root_pc(a)
        pc_b = self._root_pc(b)
        # V-I: root of V is 7 semitones above I
        return (pc_a - pc_b) % 12 == 7

    def _is_ii_v_i(self, a: str, b: str, c: str) -> bool:
        pc_a = self._root_pc(a)
        pc_b = self._root_pc(b)
        pc_c = self._root_pc(c)
        # ii-V: V is 5 semitones above ii; V-I: as above
        return (pc_b - pc_a) % 12 == 5 and (pc_b - pc_c) % 12 == 7
