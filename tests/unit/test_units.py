"""Unit tests for mass and length normalization."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from psd.units import (
    LB_TO_KG,
    MassUnit,
    NonNormalizableValueError,
    normalize_length,
    normalize_mass,
)
from tests.support import assert_close


def test_kilograms_pass_through_unchanged() -> None:
    assert normalize_mass(190.0, MassUnit.KG) == 190.0
    assert normalize_mass(190.0, "kg") == 190.0


def test_pound_conversion_uses_exact_definition() -> None:
    assert normalize_mass(1.0, MassUnit.LB) == LB_TO_KG
    assert_close(normalize_mass(225.0, "lb"), 102.05828325, rel_tol=1e-12)


def test_unknown_unit_is_refused_rather_than_assumed_kg() -> None:
    with pytest.raises(NonNormalizableValueError, match="refusing to normalize"):
        normalize_mass(100.0, "stone")


def test_unit_matching_is_case_insensitive() -> None:
    assert normalize_mass(1.0, "LB") == LB_TO_KG


def test_length_normalization() -> None:
    assert normalize_length(100.0, "cm") == 100.0
    assert_close(normalize_length(1.0, "in"), 2.54)
    assert_close(normalize_length(1.0, "m"), 100.0)


@given(
    value=st.floats(min_value=0.01, max_value=600.0, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from([MassUnit.KG, MassUnit.LB]),
)
def test_mass_normalization_is_deterministic(value: float, unit: MassUnit) -> None:
    normalized = normalize_mass(value, unit)
    assert normalized > 0.0
    assert math.isclose(normalized, normalize_mass(value, unit), rel_tol=0.0, abs_tol=0.0)


@given(value=st.floats(min_value=0.01, max_value=600.0, allow_nan=False, allow_infinity=False))
def test_kg_lb_round_trip_is_stable(value: float) -> None:
    as_kg = normalize_mass(value, MassUnit.KG)
    as_lb = as_kg / LB_TO_KG
    assert_close(normalize_mass(as_lb, MassUnit.LB), as_kg, rel_tol=1e-12)
