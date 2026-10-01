"""Mass and length normalization.

PSD accepts source values in their original unit (kilograms or pounds) and keeps
that raw value alongside a normalized kilogram value. Raw values are never
destroyed: they are the audit trail for a source that rounds to plate increments,
reports in pounds, or disagrees with the normalized value.

Normalization uses the exact international definition of the pound
(1 lb = 0.45359237 kg) so the conversion is reproducible across platforms and
library versions.

Absence is never encoded as zero: ``0 kg`` is not a missing load, it is an
implausible load, and the validation layer rejects it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = (
    "LB_TO_KG",
    "LengthUnit",
    "MassUnit",
    "NonNormalizableValueError",
    "normalize_length",
    "normalize_mass",
)

#: Exact international definition of the avoirdupoir pound in kilograms.
LB_TO_KG: Final[float] = 0.45359237

_INCH_TO_CM: Final[float] = 2.54


class NonNormalizableValueError(ValueError):
    """Raised when a value cannot be normalized without guessing."""


class MassUnit(StrEnum):
    """Units a mass may be reported in."""

    KG = "kg"
    LB = "lb"


class LengthUnit(StrEnum):
    """Units a length may be reported in."""

    CENTIMETER = "cm"
    INCH = "in"
    METER = "m"


def normalize_mass(value: float, unit: MassUnit | str) -> float:
    """Return *value* expressed in kilograms.

    Args:
        value: Mass in *unit*.
        unit: Unit the source used.

    Returns:
        The mass in kilograms.

    Raises:
        NonNormalizableValueError: The unit is not a known mass unit. Unknown
            units are never assumed to be kilograms.
    """
    resolved = _resolve_unit(unit, {MassUnit.KG.value: 1.0, MassUnit.LB.value: LB_TO_KG}, "mass")
    return float(value) * resolved


def normalize_length(value: float, unit: LengthUnit | str) -> float:
    """Return *value* expressed in centimeters.

    Args:
        value: Length in *unit*.
        unit: Unit the source used.

    Returns:
        The length in centimeters.

    Raises:
        NonNormalizableValueError: The unit is not a known length unit.
    """
    resolved = _resolve_unit(
        unit,
        {
            LengthUnit.CENTIMETER.value: 1.0,
            LengthUnit.INCH.value: _INCH_TO_CM,
            LengthUnit.METER.value: 100.0,
        },
        "length",
    )
    return float(value) * resolved


def _resolve_unit(unit: MassUnit | LengthUnit | str, factors: dict[str, float], kind: str) -> float:
    key = unit.value if isinstance(unit, StrEnum) else str(unit).strip().lower()
    factor = factors.get(key)
    if factor is None:
        msg = (
            f"Unknown {kind} unit {key!r}; refusing to normalize. Record the raw value with "
            f"a null normalized value and a missingness reason instead of assuming kilograms."
        )
        raise NonNormalizableValueError(msg)
    return factor
