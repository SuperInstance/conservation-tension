"""Chord representation, parsing, and voicing."""

from __future__ import annotations

import re
from typing import Optional

from .pitch import Pitch

# ---------------------------------------------------------------------------
# Chord-quality templates  (semitone offsets from root)
# ---------------------------------------------------------------------------
_QUALITIES: dict[str, list[int]] = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "hdim7": [0, 3, 6, 10],
    "minmaj7": [0, 3, 7, 11],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
    "add9": [0, 4, 7, 14],
}

# Quality labels derived from interval set (sorted tuple -> quality name)
_INTERVAL_TO_QUALITY: dict[tuple[int, ...], str] = {
    tuple(v): k for k, v in _QUALITIES.items()
}

# Regex for chord symbols: root + optional accidental + quality + optional alteration
# Handles: C, Cmaj7, Dm7, G7, F#m7b5, Bbmaj7, Cm, Bdim, etc.
_CHORD_RE = re.compile(
    r"^([A-Ga-g])(#|b)?"
    r"(maj7|min7|m7|maj|min|m|dim7|hdim7|dim|aug|7|sus4|sus2|add9)?"
    r"(b5|#5|b9|#9|11|13)?$"
)


def _detect_quality(intervals: list[int]) -> str:
    """Map a list of semitone intervals from root to a quality string."""
    normalized = sorted({i % 12 for i in intervals})
    t = tuple(normalized)
    if t in _INTERVAL_TO_QUALITY:
        return _INTERVAL_TO_QUALITY[t]
    # Fallback heuristic
    has_3 = 3 in normalized or 4 in normalized
    has_7 = 7 in normalized
    if not has_3:
        return "sus" if has_7 else "unknown"
    if 3 in normalized and 6 in normalized:
        if 9 in normalized:
            return "dim7"
        if 10 in normalized:
            return "hdim7"
        return "dim"
    if 4 in normalized and 8 in normalized:
        return "aug"
    if 3 in normalized:
        return "minor"
    return "major"


class Chord:
    """A musical chord built from a symbol or explicit pitches."""

    def __init__(self, name: str, pitches: Optional[list[Pitch]] = None):
        self._symbol = name.strip()
        if pitches is not None:
            self._pitches = list(pitches)
        else:
            self._pitches = self._build_pitches(self._symbol)
        if self._pitches:
            root_midi = self._pitches[0].midi
            self._intervals = sorted({p.midi - root_midi for p in self._pitches})
        else:
            self._intervals = []

    @staticmethod
    def parse(symbol: str) -> Chord:
        return Chord(symbol)

    def _build_pitches(self, symbol: str) -> list[Pitch]:
        m = _CHORD_RE.match(symbol.strip())
        if m is None:
            raise ValueError(f"Cannot parse chord symbol: {symbol!r}")
        root_letter = m.group(1).upper()
        acc = m.group(2) or ""
        quality_str = m.group(3) or ""
        alteration = m.group(4) or ""

        root_name = root_letter + acc + "4"
        root = Pitch.from_name(root_name)
        root_midi = root.midi

        intervals: list[int] = list(self._resolve_intervals(quality_str, alteration))
        pitches = []
        for iv in intervals:
            midi = root_midi + iv
            pitches.append(Pitch.from_midi(midi))
        return pitches

    @staticmethod
    def _resolve_intervals(quality_str: str, alteration: str) -> list[int]:
        """Return semitone intervals for a quality+alteration combo."""
        qmap: dict[str, list[int]] = {
            "7": _QUALITIES["dom7"],
            "maj7": _QUALITIES["maj7"],
            "min7": _QUALITIES["min7"],
            "m7": _QUALITIES["min7"],
            "m": _QUALITIES["minor"],
            "min": _QUALITIES["minor"],
            "maj": _QUALITIES["major"],
            "dim7": _QUALITIES["dim7"],
            "hdim7": _QUALITIES["hdim7"],
            "minmaj7": _QUALITIES["minmaj7"],
            "dim": _QUALITIES["dim"],
            "aug": _QUALITIES["aug"],
            "sus4": _QUALITIES["sus4"],
            "sus2": _QUALITIES["sus2"],
            "add9": _QUALITIES["add9"],
            "": _QUALITIES["major"],  # bare note name = major triad
        }
        intervals = list(qmap.get(quality_str, _QUALITIES["major"]))
        if alteration == "b5":
            if 7 in intervals:
                intervals[intervals.index(7)] = 6
            elif 8 in intervals:
                intervals[intervals.index(8)] = 7
        elif alteration == "#5":
            if 7 in intervals:
                intervals[intervals.index(7)] = 8
        elif alteration == "b9":
            intervals.append(13)
        elif alteration == "#9":
            intervals.append(15)
        return sorted(set(intervals))

    @property
    def root(self) -> Pitch:
        return self._pitches[0]

    @property
    def quality(self) -> str:
        return _detect_quality(self._intervals)

    @property
    def pitches(self) -> list[Pitch]:
        return list(self._pitches)

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def intervals(self) -> list[int]:
        return list(self._intervals)

    def tension(self, method: str = "spectral") -> float:
        from .tension import TensionMeter
        if method == "spectral":
            return TensionMeter.spectral_tension(self)
        elif method in ("voice_leading", "contextual"):
            return TensionMeter.spectral_tension(self)
        else:
            raise ValueError(f"Unknown tension method: {method!r}")

    def __repr__(self) -> str:
        return f"Chord({self._symbol!r}, pitches={self._pitches})"
