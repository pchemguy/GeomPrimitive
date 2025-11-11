def unit_circular_arc(start_deg: float = 0.0, end_deg: float = 90.0,
                      jitter_amp: float = 0.02, jitter_y: float = 0.1,
                      max_angle_step_deg: float = 20.0, min_angle_steps: int = 3,
                      rng: RNGBackend = None) -> mplPath:
    """Generate a unit circular arc as a multi-segment cubic Bezier path.
    
    Each sub-arc spans <= `max_angle_step_deg` (default 20deg) and uses the analytic
    4/3*tan(Dtheta/4) handle length. Optional additive and multiplicative jitter
    imitate human sketch irregularities.
    
    Args:
        start_deg: Starting angle in degrees.
        end_deg:   Ending angle in degrees.
        jitter_amp: Maximum additive vertex jitter (fraction of radius).
        jitter_y:   Multiplicative Y-axis jitter amplitude (~ vertical squish).
        max_angle_step_deg: Maximum angular step per Bezier segment.
        min_angle_steps: Minimum segment count, even for small spans.
        rng: Optional RNG backend (`random.Random`, `np.random.Generator`, or custom).
    
    Returns:
      matplotlib.path.Path: Composite cubic Bezier path approximating the arc.
    """
    # RNG setup ---------------------------------------------------------------
    if rng is None:
        rng = random

    # Angle normalization -----------------------------------------------------
    if start_deg is None or end_deg is None:
        start_deg = uniform(0, 270)
        end_deg = rng.uniform(start_deg + 5, 360)
    span_deg = end_deg - start_deg

    if span_deg < 1 or span_deg > 359:
        start_deg, end_deg, span_deg = 0, 360, 360
        closed = True
    else:
        closed = False

    # Segmentation ------------------------------------------------------------
    theta_steps = int(max(min_angle_steps, round(span_deg / max_angle_step_deg)))
    start, end = np.radians(start_deg), np.radians(end_deg)
    span = end - start
    step_theta = span / theta_steps
    t = 4.0 / 3.0 * np.tan(step_theta / 4.0)

    # Build control vertices --------------------------------------------------
    verts: list[PointXY] = []
    theta_beg = start
    for i in range(theta_steps):
        theta_end = theta_beg + step_theta
        cos_b, sin_b = np.cos(theta_beg), np.sin(theta_beg)
        cos_e, sin_e = np.cos(theta_end), np.sin(theta_end)

        P0 = (cos_b, sin_b)
        P1 = (cos_b - t * sin_b, sin_b + t * cos_b)
        P2 = (cos_e + t * sin_e, sin_e - t * cos_e)
        P3 = (cos_e, sin_e)

        if i == 0:
            verts.append(P0)
        verts.extend([P1, P2, P3])
        theta_beg = theta_end

    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * (3 * theta_steps)

    if closed:
        verts.append((np.nan, np.nan))
        codes.append(mplPath.CLOSEPOLY)

    verts = np.array(verts, dtype=float)

    # Y-axis multiplicative jitter --------------------------------------------
    if jitter_y:
        verts[:, 1] *= 1 - rng.uniform(0, 1) * jitter_y

    # Additive jitter ---------------------------------------------------------
    if jitter_amp:
      verts += rng.uniform(-1, 1, size=verts.shape) * jitter_amp

    return mplPath(verts, codes)
