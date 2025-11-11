def unit_triangle_path(equal_sides: int = None,
                       angle_category: int = None,
                       jitter_angle_deg: int = 5,
                       base_angle: int = None,
                       rng: RNGBackend = None,
                      ) -> mplPath:
    """Generates vertices of triangle inscribed into a unit circle.

    Arguments:
        "equal_sides":      1, 2, or 3.
        "angle_category":   This value is compared with 90 to determine
                            requested triangle (actual value is not used):
                            <90 - ACUTE
                            =90 - RIGHT
                            >90 - OBTUSE
    """
    # --- RNG ----------------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)
    normal3s = getattr(rng, "normal3s",
                   lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))))

    if not equal_sides:
        equal_sides = rng.choice((1, 2, 3))
    if not equal_sides in (1, 2, 3):
        raise ValueError(
            f"equal_sides must be an integer in [1, 3].\n"
            f"Received type: {type(equal_sides).__name__}; value: {equal_sides}."
        )
    if not angle_category:
        angle_category = rng.choice((60, 90, 120))
    if not isinstance(angle_category, (int, float)):
        raise TypeError(
            f"angle_category must be ot type integer or float.\n"
            f"Received type: {type(angle_category).__name__}; value: {angle_category}."
        )

    if equal_sides == 3:
        thetas = [90, -30, 210]
    else:
        top_offset = (
            0 if equal_sides > 1 else rng.choice([-1, 1]) *
            rng.uniform(jitter_angle_deg, 90 - jitter_angle_deg)
        )
        base_offset = (
            ((angle_category > 90) - (angle_category < 90)) *
            rng.uniform(jitter_angle_deg, 90 - jitter_angle_deg)
        )
        thetas = [90 + top_offset, 0 + base_offset, 180 - base_offset]

    top_jitter = normal3s() * jitter_angle_deg
    thetas[0] += top_jitter

    if not isinstance(base_angle, (int, float)):
        base_angle = rng.uniform(-90, 90)
    else:
        base_angle += normal3s() * jitter_angle_deg
    thetas = [(theta + base_angle) for theta in thetas]
    thetas = [math.radians(((theta + 180) % 360) - 180) for theta in thetas]
    verts = [(math.cos(theta_rad), math.sin(theta_rad)) for theta_rad in thetas]
    verts.append(verts[0])
    codes = [mplPath.MOVETO, mplPath.LINETO, mplPath.LINETO, mplPath.CLOSEPOLY]

    return mplPath(verts, codes)
