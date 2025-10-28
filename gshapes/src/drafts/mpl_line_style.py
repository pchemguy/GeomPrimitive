import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


def pattern_to_linestyle(s: str) -> Tuple[int, Tuple[int, ...]]:
    """
    Convert a symbolic pattern string into a Matplotlib-compatible linestyle.

    Supports both explicit symbolic definitions and random generation.

    Rules:
        - ' ' : off (gap) of 1 unit
        - '_' : off (gap) of 4 units
        - '-' : on  (dash) of 4 units
        - '.' : on  (dash) of 1 unit
        - Leading spaces are ignored (no offset)
        - If the pattern does not end with an off-segment,
          an additional off-segment equal to the last dash length is appended.
        - If the input string is empty, a random pattern is generated.

    Random pattern generation:
        - Pattern length = 2N, where N in [1, 5]
        - Each on/off length in [1, 5]

    Args:
        s (str): Pattern string containing ' ', '_', '-', and '.' characters.

    Returns:
        Tuple[int, Tuple[int, ...]]: A Matplotlib linestyle tuple `(offset, pattern)`.

    Example:
        >>> pattern_to_linestyle("--__-.  ")
        (0, (8, 8, 5, 2))
        >>> pattern_to_linestyle("_--__-.")
        (0, (8, 8, 5, 5))
        >>> pattern_to_linestyle("")
        (0, (3, 5, 2, 4))
    """
    # Handle the empty-string case by generating a random pattern
    if not s.strip():
        n_segments: int = random.randint(1, 5)  # number of on/off pairs
        pattern: Tuple[int, ...] = tuple(random.randint(1, 5) for _ in range(2 * n_segments))
        return (0, pattern)

    # Character mapping: defines whether each symbol is "on" or "off" and its length
    mapping: dict[str, Tuple[str, int]] = {
        " ": ("off", 1),
        "_": ("off", 4),
        "-": ("on", 4),
        ".": ("on", 1),
    }

    # Remove leading spaces, as they do not contribute to the offset
    s = s.lstrip(" ")
    if not s:
        raise ValueError("Pattern is empty after trimming leading spaces.")

    # Sequentially build the pattern by merging consecutive segments of the same type
    segments: List[int] = []
    last_type: str | None = None
    last_length: int = 0

    for ch in s:
        if ch not in mapping:
            raise ValueError(f"Invalid character in pattern: {ch!r}")
        seg_type, seg_len = mapping[ch]

        # If same type as previous, extend the segment
        if seg_type == last_type and segments:
            segments[-1] += seg_len
        else:
            segments.append(seg_len)
            last_type = seg_type

        last_length = seg_len  # Track the last segment length for trailing adjustment

    # Ensure the pattern alternates between on/off - must have an even number of entries
    if len(segments) % 2 != 0:
        # Append a final "off" segment equal to the last dash length
        segments.append(last_length)

    return (0, tuple(segments))


def plot_pattern_preview(patterns: List[str]) -> None:
    """
    Visualize multiple symbolic or random dash patterns for visual inspection.

    Each pattern string is converted into a Matplotlib linestyle and plotted
    on a separate horizontal line.

    Args:
        patterns (List[str]): List of symbolic or empty pattern strings.

    Example:
        >>> plot_pattern_preview(["--__-.  ", "_--__-.", "", "", ""])
    """
    x: np.ndarray = np.linspace(0, 10, 200)
    y0: np.ndarray = np.zeros_like(x)

    plt.figure(figsize=(8, len(patterns) * 1.2))

    for i, pattern in enumerate(patterns):
        linestyle: Tuple[int, Tuple[int, ...]] = pattern_to_linestyle(pattern)
        label: str = f"{pattern!r} - {linestyle[1]}"
        plt.plot(x, y0 + i, linestyle=linestyle, linewidth=3, label=label)

    plt.yticks([])
    plt.title("Custom Matplotlib Line Patterns (Explicit + Random)", fontsize=12)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Fixed test patterns demonstrating explicit symbolic logic
    test_patterns: List[str] = [
        "--__-.  ",
        "_--__-.",
        "  -_.",
        "--_..__",
        ".-.-.- ",
    ]

    # Add 5 random patterns (empty string - random generation)
    random_patterns: List[str] = [""] * 5

    # Display both fixed and random dash styles
    plot_pattern_preview(test_patterns + random_patterns)
