#!/usr/bin/env python3
"""Extract the three kagome bands from a COMSOL CSV export.

The input file contains three complex-valued eigenfrequencies for every value
of ``b`` along the Gamma-K-M-Gamma path.  Only the real part of each frequency
is retained.  Bands are matched between adjacent b values by frequency
continuity, so their labels remain consistent through crossings.
"""

from __future__ import annotations

import argparse
import csv
from itertools import permutations
from pathlib import Path
from typing import Sequence


NUMBER_OF_BANDS = 3
FREQUENCY_UNIT_TO_HZ = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[3] / "data" / "simulation" / "kagome_bands.csv"
)


def _real_part(value: str) -> float:
    """Convert a real or COMSOL-style complex number to its real part."""
    try:
        return complex(value.strip().replace("i", "j")).real
    except ValueError as error:
        raise ValueError(f"Could not parse frequency value {value!r}") from error


def _frequency_scale_to_hz(column_name: str) -> float:
    """Return the Hz conversion factor encoded in a frequency column name."""
    normalized_name = column_name.casefold().replace(" ", "")
    for unit, scale in FREQUENCY_UNIT_TO_HZ.items():
        if f"({unit})" in normalized_name:
            return scale
    supported_units = ", ".join(FREQUENCY_UNIT_TO_HZ)
    raise ValueError(
        f"Could not identify a frequency unit in column {column_name!r}; "
        f"supported units are {supported_units}"
    )


def _read_frequency_rows(input_path: Path) -> list[tuple[float, float]]:
    """Read (b, frequency in Hz) pairs from a COMSOL CSV."""
    header: list[str] | None = None
    b_index: int | None = None
    frequency_index: int | None = None
    frequency_scale_to_hz: float | None = None
    rows: list[tuple[float, float]] = []

    with input_path.open(newline="", encoding="utf-8-sig") as csv_file:
        for raw_row in csv.reader(csv_file):
            if not raw_row:
                continue

            if raw_row[0].lstrip().startswith("%"):
                candidate = [cell.strip().lstrip("% ") for cell in raw_row]
                frequency_columns = [
                    index
                    for index, name in enumerate(candidate)
                    if "frequency" in name.casefold()
                ]
                if frequency_columns:
                    header = candidate
                    frequency_index = frequency_columns[-1]
                    frequency_scale_to_hz = _frequency_scale_to_hz(
                        header[frequency_index]
                    )
                    preceding_b_columns = [
                        index
                        for index, name in enumerate(candidate[:frequency_index])
                        if name.casefold() == "b"
                    ]
                    if not preceding_b_columns:
                        raise ValueError(
                            f"No b column precedes {header[frequency_index]!r} in "
                            f"{input_path}"
                        )
                    b_index = preceding_b_columns[-1]
                continue

            if (
                header is None
                or b_index is None
                or frequency_index is None
                or frequency_scale_to_hz is None
            ):
                raise ValueError(
                    f"Could not find the b and Frequency columns in {input_path}"
                )
            if len(raw_row) <= max(b_index, frequency_index):
                raise ValueError(f"Incomplete CSV row in {input_path}: {raw_row}")

            frequency_hz = (
                _real_part(raw_row[frequency_index]) * frequency_scale_to_hz
            )
            rows.append((float(raw_row[b_index]), frequency_hz))

    if not rows:
        raise ValueError(f"No band data found in {input_path}")
    return rows


def _track_bands(
    b_values: Sequence[float], frequencies_by_b: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Assign frequencies using linear continuation from preceding b values."""
    first_frequencies = sorted(frequencies_by_b[0])
    tracked = [[frequency] for frequency in first_frequencies]

    for index, frequencies in enumerate(frequencies_by_b[1:], start=1):
        candidates = sorted(frequencies)
        if index == 1:
            predicted = [band[-1] for band in tracked]
        else:
            previous_step = b_values[index - 1] - b_values[index - 2]
            current_step = b_values[index] - b_values[index - 1]
            if previous_step <= 0 or current_step <= 0:
                raise ValueError("b values must be strictly increasing")
            step_ratio = current_step / previous_step
            predicted = [
                band[-1] + step_ratio * (band[-1] - band[-2]) for band in tracked
            ]

        assignment = min(
            permutations(candidates),
            key=lambda values: sum(
                abs(current - expected)
                for current, expected in zip(values, predicted)
            ),
        )
        for band, frequency in zip(tracked, assignment):
            band.append(frequency)

    return tracked


def extract_band_frequencies(input_path: str | Path) -> tuple[list[float], list[list[float]]]:
    """Return b values and three frequency-continuous bands in Hz.

    The returned ``bands`` list has shape ``(3, len(b_values))``.  At the first
    b value the bands are ordered from lowest to highest frequency.
    """
    input_path = Path(input_path)
    grouped: dict[float, list[float]] = {}
    for b_value, frequency in _read_frequency_rows(input_path):
        grouped.setdefault(b_value, []).append(frequency)

    invalid_counts = {
        b_value: len(frequencies)
        for b_value, frequencies in grouped.items()
        if len(frequencies) != NUMBER_OF_BANDS
    }
    if invalid_counts:
        details = ", ".join(
            f"b={b_value:g}: {count}" for b_value, count in sorted(invalid_counts.items())
        )
        raise ValueError(
            f"Expected {NUMBER_OF_BANDS} frequencies at every b value; found {details}"
        )

    b_values = sorted(grouped)
    bands = _track_bands(b_values, [grouped[b_value] for b_value in b_values])
    return b_values, bands


def write_band_csv(
    output_path: str | Path,
    b_values: Sequence[float],
    bands: Sequence[Sequence[float]],
) -> None:
    """Write b and the three extracted band frequencies to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["b"]
            + [
                f"band_{band_number}_frequency_Hz"
                for band_number in range(1, NUMBER_OF_BANDS + 1)
            ]
        )
        for index, b_value in enumerate(b_values):
            writer.writerow(
                [format(b_value, ".12g")]
                + [format(band[index], ".17g") for band in bands]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract three frequency-continuous kagome bands versus b."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"COMSOL CSV input (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output CSV (default: <input stem>_extracted.csv next to the input)",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_name(f"{args.input.stem}_extracted.csv")
    b_values, bands = extract_band_frequencies(args.input)
    write_band_csv(output_path, b_values, bands)
    print(f"Wrote {len(b_values)} b points and {len(bands)} bands to {output_path}")


if __name__ == "__main__":
    main()
