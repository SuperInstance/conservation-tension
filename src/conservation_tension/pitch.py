"""Pitch, interval, cents, and frequency utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# Note names to semitone offset from C
_NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Regex: optional accidental (b or #), optional octave (default 4)
_NAME_RE = re.compile(r"^([A-Ga-g])(#|b)?(\d)?$")


@dataclass(frozen=True, slots=True)
class Pitch:
    """A musical pitch with name, MIDI number, and frequency."""

    name: str
    midi: int
    frequency: float

    _A4_MIDI: ClassVar[int] = 69
    _A4_FREQ: ClassVar[float] = 440.0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @staticmethod
    def from_name(name: str) -> Pitch:
        m = _NAME_RE.match(name.strip())
        if m is None:
            raise ValueError(f"Invalid pitch name: {name!r}")
        letter, acc, oct_str = m.group(1).upper(), m.group(2) or "", m.group(3)
        octave = int(oct_str) if oct_str else 4
        key = letter + acc
        if key not in _NOTE_MAP:
            raise ValueError(f"Unknown note: {key}")
        semitone = _NOTE_MAP[key]
        midi = (octave + 1) * 12 + semitone
        freq = Pitch._midi_to_freq(midi)
        return Pitch(name=name, midi=midi, frequency=freq)

    @staticmethod
    def from_midi(midi: int) -> Pitch:
        freq = Pitch._midi_to_freq(midi)
        octave = midi // 12 - 1
        semitone = midi % 12
        # Find canonical name
        _SEMITONE_TO_NAME = {
            0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
            6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
        }
        name = f"{_SEMITONE_TO_NAME[semitone]}{octave}"
        return Pitch(name=name, midi=midi, frequency=freq)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def interval_to(self, other: Pitch) -> float:
        """Return interval in cents between self and *other*."""
        if self.frequency == 0:
            return 0.0
        return 1200.0 * np.log2(other.frequency / self.frequency)

    def harmonics(self, n: int = 8) -> np.ndarray:
        """Return first *n* harmonics (including fundamental) as frequencies."""
        return np.array([self.frequency * (k + 1) for k in range(n)])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _midi_to_freq(midi: int) -> float:
        return Pitch._A4_FREQ * 2.0 ** ((midi - Pitch._A4_MIDI) / 12.0)

    def __repr__(self) -> str:
        return f"Pitch({self.name}, midi={self.midi}, freq={self.frequency:.2f})"
