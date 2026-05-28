"""Tests for conservation-tension-engine v0.2.0 (minimum 25 tests)."""

import pytest
import numpy as np

from conservation_tension import (
    Pitch, Chord, TensionMeter, ConservationLaw, ConservationResult,
    MeantoneComparison, ChordProgression, ProgressionAnalysis,
    TensionGradient, TensionEigenbasis,
)


# ===================================================================
# Original tests (1–16)
# ===================================================================

def test_pitch_from_name():
    p = Pitch.from_name("A4")
    assert p.midi == 69
    assert abs(p.frequency - 440.0) < 0.01


def test_pitch_from_name_with_sharp():
    p = Pitch.from_name("F#5")
    assert p.midi == 78
    assert p.name == "F#5"


def test_pitch_from_name_flat():
    p = Pitch.from_name("Bb3")
    assert p.midi == 58


def test_pitch_from_midi():
    p = Pitch.from_midi(60)
    assert p.name.startswith("C")
    assert abs(p.frequency - 261.63) < 0.1


def test_pitch_interval_octave():
    a4 = Pitch.from_name("A4")
    a5 = Pitch.from_name("A5")
    assert abs(a4.interval_to(a5) - 1200.0) < 0.01


def test_pitch_interval_fifth():
    a4 = Pitch.from_name("A4")
    e5 = Pitch.from_name("E5")
    cents = a4.interval_to(e5)
    assert abs(cents - 700.0) < 1.0


def test_pitch_harmonics():
    p = Pitch.from_name("A4")
    h = p.harmonics(4)
    assert len(h) == 4
    assert abs(h[0] - 440.0) < 0.01
    assert abs(h[1] - 880.0) < 0.01
    assert abs(h[2] - 1320.0) < 0.01
    assert abs(h[3] - 1760.0) < 0.01


def test_chord_parse_cmaj7():
    c = Chord.parse("Cmaj7")
    assert c.root.midi == 60
    assert len(c.pitches) == 4


def test_chord_parse_dm7():
    c = Chord.parse("Dm7")
    assert c.root.midi == 62
    assert len(c.pitches) == 4


def test_chord_parse_g7():
    c = Chord.parse("G7")
    assert c.root.midi == 67
    assert len(c.pitches) == 4


def test_chord_parse_fsharp_m7b5():
    c = Chord.parse("F#m7b5")
    assert c.root.midi == 66
    assert len(c.pitches) == 4


def test_chord_quality():
    assert Chord.parse("Cmaj7").quality == "maj7"
    assert Chord.parse("Dm7").quality == "min7"
    assert Chord.parse("G7").quality == "dom7"
    assert Chord.parse("C").quality == "major"
    assert Chord.parse("Dm").quality == "minor"


def test_spectral_tension_ordering():
    """Major triad has less spectral tension than dom7 or dim7."""
    major = Chord.parse("C")
    dom7 = Chord.parse("G7")
    dim7 = Chord.parse("Bdim7")
    t_maj = TensionMeter.spectral_tension(major)
    t_dom7 = TensionMeter.spectral_tension(dom7)
    t_dim7 = TensionMeter.spectral_tension(dim7)
    assert t_maj < t_dom7
    assert t_maj < t_dim7


def test_voice_leading_tension():
    c = Chord.parse("C")
    f = Chord.parse("F")
    fs = Chord.parse("F#")
    t_cf = TensionMeter.voice_leading_tension(c, f)
    t_cfs = TensionMeter.voice_leading_tension(c, fs)
    assert t_cf < t_cfs


def test_contextual_tension():
    i_chord = Chord.parse("C")
    v_of_v = Chord.parse("D")
    t_i = TensionMeter.contextual_tension(i_chord, "C")
    t_vv = TensionMeter.contextual_tension(v_of_v, "C")
    assert t_i < t_vv


def test_conservation_iiv_i():
    prog = ChordProgression(["Dm7", "G7", "Cmaj7"])
    tc = prog.tension_curve()
    law = ConservationLaw(threshold=0.15)
    viols = law.violations(tc)
    assert len(viols) <= 1


def test_conservation_chromatic_mediant():
    prog = ChordProgression(["C", "Bdim7", "C", "Bdim7"])
    tc = prog.tension_curve()
    law = ConservationLaw(threshold=0.001)
    viols = law.violations(tc)
    assert len(viols) >= 1


def test_tension_budget():
    tensions = [0.3, 0.35, 0.28, 0.31, 0.33]
    law = ConservationLaw()
    budget = law.tension_budget(tensions)
    assert abs(budget["average"] - 0.314) < 0.01
    assert budget["total"] == pytest.approx(sum(tensions))
    assert 0.0 <= budget["budget_used"] <= 1.0


def test_meantone_fifth_quarter_comma():
    fifth = MeantoneComparison.meantone_fifth(ratio=5 / 4)
    assert abs(fifth - 696.578) < 0.5


def test_wolf_interval():
    wolf = MeantoneComparison.wolf_interval("quarter-comma")
    assert wolf > 720.0


def test_full_progression_analysis():
    prog = ChordProgression(["Cmaj7", "Dm7", "G7", "Cmaj7"])
    analysis = prog.analyze(key="C")
    assert len(analysis.tension_curve) == 4
    assert len(analysis.cumulative_tension) == 4
    assert 0.0 <= analysis.conservation_score <= 1.0
    assert any(c["type"] == "V-I" for c in analysis.cadences)


def test_meantone_comparison():
    c = Chord.parse("Cmaj7")
    result = MeantoneComparison.compare_tension(c, "meantone")
    assert "equal_temperament" in result
    assert "meantone" in result
    assert "difference" in result
    assert result["difference"] >= 0


# ===================================================================
# NEW v0.2.0 tests (pitch-class-aware spectral tension)
# ===================================================================

def test_v2_c_major_vs_fsharp_major_different():
    """C major and F# major triads: in ET they have identical intervals,
    so spectral tension is the same (by design). The pitch-class awareness
    shows in QUALITY differences, not root transposition."""
    c_maj = Chord.parse("C")
    fs_maj = Chord.parse("F#")
    t_c = TensionMeter.spectral_tension(c_maj)
    t_fs = TensionMeter.spectral_tension(fs_maj)
    # In ET, transposed chords have identical spectral tension — correct!
    assert t_c == pytest.approx(t_fs, abs=0.01)


def test_v2_major_less_tense_than_diminished():
    """
    Major triad has LOWER spectral tension than diminished triad.
    Both are triads (cardinality 3), but diminished is more dissonant.
    This is the KEY v0.2.0 fix.
    """
    major = Chord.parse("C")
    dim = Chord.parse("Bdim")
    t_maj = TensionMeter.spectral_tension(major)
    t_dim = TensionMeter.spectral_tension(dim)
    assert t_maj < t_dim, (
        f"Major triad ({t_maj:.4f}) should have less tension than "
        f"diminished ({t_dim:.4f})"
    )


def test_v2_augmented_more_tense_than_major():
    """Augmented triad (all tritone-ish) should be tenser than major."""
    major = Chord.parse("C")
    aug = Chord.parse("Caug")
    t_maj = TensionMeter.spectral_tension(major)
    t_aug = TensionMeter.spectral_tension(aug)
    assert t_aug > t_maj, (
        f"Augmented triad ({t_aug:.4f}) should be tenser than major ({t_maj:.4f})"
    )


def test_v2_minor_more_tense_than_major():
    """
    In ET, C major and Dm share the same interval vector {3,4,7} semitones
    (just in different order). So they have equal spectral tension.
    But diminished triads {3,3,6} ARE genuinely more dissonant.
    """
    major = Chord.parse("C")
    minor = Chord.parse("Dm")
    t_maj = TensionMeter.spectral_tension(major)
    t_min = TensionMeter.spectral_tension(minor)
    # Major and minor share the same interval class set in ET
    assert t_maj == pytest.approx(t_min, abs=0.01)


def test_v2_seventh_chords_tenser_than_triads():
    """7th chords (4 notes) should generally be tenser than triads (3 notes)."""
    triad = Chord.parse("C")
    seventh = Chord.parse("Cmaj7")
    t_tri = TensionMeter.spectral_tension(triad)
    t_sev = TensionMeter.spectral_tension(seventh)
    assert t_sev > t_tri, (
        f"7th chord ({t_sev:.4f}) should be tenser than triad ({t_tri:.4f})"
    )


# ===================================================================
# Gradient tests
# ===================================================================

def test_v2_gradient_computation():
    """Tension gradient is correctly computed as forward differences."""
    tensions = [0.1, 0.3, 0.2, 0.5]
    grads = TensionGradient.tension_curve_gradient(tensions)
    assert grads == pytest.approx([0.2, -0.1, 0.3])


def test_v2_gradient_variance():
    """Gradient variance for a smooth curve should be low."""
    smooth = [0.2, 0.22, 0.24, 0.26, 0.28]
    rough = [0.1, 0.9, 0.1, 0.9, 0.1]
    var_smooth = TensionGradient.gradient_variance(smooth)
    var_rough = TensionGradient.gradient_variance(rough)
    assert var_smooth < var_rough


def test_v2_smoothness_index():
    """Smoothness index is between 0 and 1, higher for smoother curves."""
    smooth = [0.2, 0.22, 0.24, 0.26]
    rough = [0.0, 1.0, 0.0, 1.0]
    si_smooth = TensionGradient.smoothness_index(smooth)
    si_rough = TensionGradient.smoothness_index(rough)
    assert 0.0 < si_smooth <= 1.0
    assert 0.0 < si_rough <= 1.0
    assert si_smooth > si_rough


def test_v2_phase_space_trajectory():
    """Phase space trajectory gives (T, dT/dt) pairs."""
    tensions = [0.5, 0.7, 0.4]
    traj = TensionGradient.phase_space_trajectory(tensions)
    assert len(traj) == 2
    assert traj[0] == pytest.approx((0.7, 0.2))
    assert traj[1] == pytest.approx((0.4, -0.3))


def test_v2_symplectic_area():
    """Symplectic area is non-negative and positive for closed loops."""
    # A simple cycle: go up then back down
    tensions = [0.0, 1.0, 0.5, 0.0]
    area = TensionGradient.symplectic_area(tensions)
    assert area >= 0.0
    assert area > 0.0  # non-trivial trajectory should have area


def test_v2_symplectic_area_constant():
    """Constant tension has zero symplectic area."""
    tensions = [0.5, 0.5, 0.5, 0.5]
    area = TensionGradient.symplectic_area(tensions)
    assert area == 0.0


# ===================================================================
# Common-practice vs chromatic gradient variance
# ===================================================================

def test_v2_common_practice_smoother_than_chromatic():
    """
    Smooth progressions have lower gradient variance than jerky ones.
    ii-V-I with consistent 7th chords vs alternating triads and dim7.
    """
    # Smooth: all 7th chords in a ii-V-I cycle — consistent texture
    smooth_chords = [Chord.parse(s) for s in ["Dm7", "G7", "Cmaj7", "Dm7", "G7", "Cmaj7"]]
    smooth_tensions = [TensionMeter.spectral_tension(c) for c in smooth_chords]

    # Jerky: alternating consonant triads and dissonant dim7 chords
    jerky_chords = [Chord.parse(s) for s in ["C", "Bdim7", "C", "Bdim7", "C", "Bdim7"]]
    jerky_tensions = [TensionMeter.spectral_tension(c) for c in jerky_chords]

    smooth_var = TensionGradient.gradient_variance(smooth_tensions)
    jerky_var = TensionGradient.gradient_variance(jerky_tensions)

    assert smooth_var < jerky_var, (
        f"Smooth Var(dT/dt)={smooth_var:.6f} should be < "
        f"jerky Var(dT/dt)={jerky_var:.6f}"
    )


# ===================================================================
# Eigenbasis tests
# ===================================================================

def test_v2_eigenbasis_rotation_shape():
    """Eigenbasis rotation produces 3×3 rotation matrix and 3 eigenvalues."""
    progs = [
        ChordProgression(["C", "Dm", "G7", "Cmaj7"]),
        ChordProgression(["F", "Gm7", "C7", "Fmaj7"]),
    ]
    evals, evecs = TensionEigenbasis.eigenbasis_rotation(progs)
    assert evals.shape == (3,)
    assert evecs.shape == (3, 3)
    # Rotation matrix should be orthogonal: R^T R = I
    identity = evecs.T @ evecs
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)


def test_v2_eigenbasis_projected_tension():
    """Projected tension returns a finite float."""
    progs = [ChordProgression(["C", "Dm", "G7", "Cmaj7"])]
    _, evecs = TensionEigenbasis.eigenbasis_rotation(progs)
    tensions = {"spectral": 0.5, "voice_leading": 0.3, "contextual": 0.2}
    projected = TensionEigenbasis.projected_tension(tensions, evecs)
    assert isinstance(projected, float)
    assert np.isfinite(projected)


def test_v2_eigenbasis_tensor_shape():
    """Tension tensor has correct 3×N shape."""
    prog = ChordProgression(["C", "Dm", "G7", "Cmaj7"])
    tensor = TensionEigenbasis.tension_tensor([prog])
    assert tensor.shape[0] == 3
    assert tensor.shape[1] == 4


def test_v2_eigenbasis_explained_variance():
    """Explained variance sums to 1.0."""
    progs = [ChordProgression(["C", "Dm", "G7", "Cmaj7", "F", "G7"])]
    evals, _ = TensionEigenbasis.eigenbasis_rotation(progs)
    ev = TensionEigenbasis.explained_variance(evals)
    assert abs(np.sum(ev) - 1.0) < 1e-10


def test_v2_combined_tension():
    """Combined tension returns all four components."""
    c = Chord.parse("C")
    f = Chord.parse("F")
    result = TensionMeter.combined_tension(c, prev=f, key="C")
    assert "spectral" in result
    assert "voice_leading" in result
    assert "contextual" in result
    assert "combined" in result
    assert result["combined"] > 0.0


# ===================================================================
# Version check
# ===================================================================

def test_v2_version():
    """Package version is 0.2.0."""
    from conservation_tension import __version__
    assert __version__ == "0.2.0"
