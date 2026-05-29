# conservation-tension

**v0.2.0** — Pitch-class-aware harmonic tension with Tenney height dissonance, tension gradients, phase-space analysis, and eigenbasis rotation.

## What's New in v0.2.0

- **Pitch-class-aware spectral tension**: Uses Tenney height (log₂(n·d)) to compute dissonance for each interval pair in a chord. Major triads now correctly have lower tension than diminished triads.
- **Combined tension metric**: Weighted combination of spectral (45%), voice-leading (30%), and contextual (25%) tensions.
- **Tension gradient**: Built-in dT/dt computation via `TensionGradient` — gradient variance, smoothness index, phase-space trajectories.
- **Phase-space view**: `(T, dT/dt)` trajectories with symplectic area computation via the shoelace formula.
- **Eigenbasis rotation**: `TensionEigenbasis` rotates the 3-component tension tensor into its principal components for conservation analysis.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from conservation_tension import Chord, TensionMeter, TensionGradient, TensionEigenbasis
from conservation_tension import ChordProgression

# Pitch-class-aware spectral tension
major = Chord.parse("C")
dim = Chord.parse("Bdim7")
print(f"Major: {TensionMeter.spectral_tension(major):.4f}")  # lower
print(f"Dim7:  {TensionMeter.spectral_tension(dim):.4f}")    # higher

# Combined tension with all components
result = TensionMeter.combined_tension(major, key="C")
print(result)  # {'spectral': ..., 'voice_leading': ..., 'contextual': ..., 'combined': ...}

# Tension gradient analysis
prog = ChordProgression(["Dm7", "G7", "Cmaj7"])
tensions = [TensionMeter.spectral_tension(c) for c in prog.chords]
print(f"Gradient variance: {TensionGradient.gradient_variance(tensions):.6f}")
print(f"Smoothness index:  {TensionGradient.smoothness_index(tensions):.4f}")
print(f"Symplectic area:   {TensionGradient.symplectic_area(tensions):.4f}")

# Eigenbasis rotation
evals, evecs = TensionEigenbasis.eigenbasis_rotation([prog])
print(f"Eigenvalues: {evals}")
print(f"Explained variance: {TensionEigenbasis.explained_variance(evals)}")
```

## Architecture

- `pitch.py` — Pitch representation with MIDI, frequency, and cents
- `chord.py` — Chord parsing and quality detection
- `tension.py` — Pitch-class-aware spectral tension (Tenney height), voice-leading, contextual, combined
- `gradient.py` — Tension gradient dT/dt, variance, smoothness, phase space, symplectic area
- `eigenbasis.py` — PCA rotation of tension components for conservation analysis
- `conservation.py` — Conservation law checking with violation detection
- `meantone.py` — Meantone temperament comparison
- `progression.py` — ChordProgression with full analysis

## Testing

```bash
pytest tests/ -v
```

40 tests covering all modules including pitch-class-aware tension ordering, gradient computation, phase space, and eigenbasis rotation.

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
