import pytest
from gshapes.core import create_circle

def test_create_circle():
    """Tests the create_circle function."""
    circle = create_circle(10)
    assert circle[0] == 10
    assert circle[1] > 314

def test_create_circle_invalid_radius():
    """Tests that create_circle raises an error for invalid radius."""
    with pytest.raises(ValueError):
        create_circle(-1)
