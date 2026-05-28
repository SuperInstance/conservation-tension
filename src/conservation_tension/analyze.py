"""High-level analysis of a piece or passage."""

from __future__ import annotations

from .progression import ChordProgression, ProgressionAnalysis


def analyze(
    chords: list[str] | str,
    key: str = "C",
    threshold: float = 0.1,
) -> ProgressionAnalysis:
    """Analyze a chord progression (list of symbols or pipe-separated string).

    Example::

        result = analyze("Cmaj7 | Dm7 | G7 | Cmaj7", key="C")
        print(result.tension_curve)
    """
    if isinstance(chords, str):
        chords = [c.strip() for c in chords.split("|")]
    prog = ChordProgression(chords)
    return prog.analyze(key=key)
