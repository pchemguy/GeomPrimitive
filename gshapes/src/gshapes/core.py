import numpy as np

def create_circle(radius: float) -> np.ndarray:
    """Creates a circle of a given radius."""
    if radius <= 0:
        raise ValueError("Radius must be positive")
    return np.array([radius, np.pi * radius**2])
