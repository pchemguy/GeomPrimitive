# https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, find_peaks


def analyze_grid_centers(centers, optimize_axis='x', search_angle=0.0):
    """
    Robustly determines the grid period and rotation from a set of point centers.
    
    Args:
        centers: (N, 2) numpy array of [x, y] coordinates.
        optimize_axis: 'x' to find vertical line spacing (project onto X),
                       'y' to find horizontal line spacing (project onto Y).
        search_angle: Expected rotation (degrees). Search will occur +/- 10 deg around this.

    Returns:
        dict: {
            'period': float (pixels),
            'angle': float (degrees),
            'rms_error': float (pixel error),
            'confidence': str ('HIGH', 'MODERATE', 'LOW')
        }
    """
    # 1. OPTIMAL ROTATION
    # We find the angle that maximizes the sharpness of the grid projection.
    best_angle = _get_optimal_angle(centers, optimize_axis, search_angle)
    
    # 2. PROJECTION
    # Rotate points to the corrected frame to get a 1D signal
    theta = np.radians(best_angle)
    c, s = np.cos(theta), np.sin(theta)
    
    if optimize_axis == 'x':
        # Project onto X (Vertical lines) -> x' = x*c - y*s
        proj = centers[:, 0] * c - centers[:, 1] * s
    else:
        # Project onto Y (Horizontal lines) -> y' = x*s + y*c
        proj = centers[:, 0] * s + centers[:, 1] * c
        
    # 3. COARSE PERIOD (Global Autocorrelation)
    # This solves the "dashed line" gap issue. It looks for the global "beat"
    # rather than measuring individual gaps.
    
    # Create 1D density signal (1 pixel bins)
    p_min, p_max = proj.min(), proj.max()
    bins = int(p_max - p_min)
    if bins < 10: return {'error': 'Data span too small'}
    
    hist, _ = np.histogram(proj, bins=bins)
    
    # Auto-correlate
    corr = correlate(hist, hist, mode='full')
    lags = np.arange(len(corr)) - (len(corr)//2)
    
    # Isolate positive lags (ignore lag 0 which is the signal matching itself)
    # We assume period is at least 10px to avoid noise at lag 1-5
    mask = (lags > 10) & (lags < len(lags)//2)
    valid_lags = lags[mask]
    valid_corr = corr[mask]
    
    if len(valid_lags) == 0:
        return {'period': 0, 'confidence': 'FAIL', 'rms_error': 999}

    # Find peaks in correlation
    # Prominence helps ignore small noise bumps
    peaks, _ = find_peaks(valid_corr, prominence=np.max(valid_corr)*0.1)
    
    if len(peaks) == 0:
        # Fallback: just take max
        coarse_period = valid_lags[np.argmax(valid_corr)]
    else:
        # The first strong peak is the fundamental frequency
        # (Sometimes the 2nd peak is higher due to signal overlap, but 1st is the period)
        # We take the strongest peak that isn't absurdly close to 0
        sorted_peaks = peaks[np.argsort(valid_corr[peaks])[::-1]] # Sort by height
        best_peak_idx = sorted_peaks[0]
        coarse_period = valid_lags[best_peak_idx]

    # 4. FINE REFINEMENT (Phase Minimization)
    # Autocorrelation gives integer resolution (e.g. 44px).
    # Real grids are sub-pixel (e.g. 44.3px).
    # We refine by minimizing the "fuzziness" of the modulo.
    
    def phase_variance(p):
        if p <= 0: return 1e9
        # Modulo
        phase = proj % p
        # Calculate wrapping standard deviation
        # (Points at 0.1 and 43.9 are actually close if period is 44)
        # We shift the phase to be centered at 0
        phase_centered = (phase - np.median(phase))
        phase_centered[phase_centered > p/2] -= p
        phase_centered[phase_centered < -p/2] += p
        return np.std(phase_centered)

    # Search in a tight +/- 10% window around the coarse guess
    res = minimize_scalar(
        phase_variance, 
        bounds=(coarse_period * 0.9, coarse_period * 1.1), 
        method='bounded'
    )
    fine_period = res.x
    final_rms = res.fun # The std dev we minimized IS the RMS error
    
    # 5. VERDICT
    if final_rms < 1.0: confidence = "HIGH"
    elif final_rms < 2.0: confidence = "MODERATE"
    else: confidence = "LOW"

    return {
        'period': round(fine_period, 4),
        'angle': round(best_angle, 4),
        'rms_error': round(final_rms, 4),
        'confidence': confidence,
        'coarse_guess': coarse_period
    }

def _get_optimal_angle(centers, axis, ref_angle):
    # Center data
    pts = centers - centers.mean(axis=0)
    x, y = pts[:, 0], pts[:, 1]
    
    def objective(deg):
        theta = np.radians(deg)
        c, s = np.cos(theta), np.sin(theta)
        if axis == 'x':
            # Project to X
            p = x*c - y*s
        else:
            # Project to Y
            p = x*s + y*c
        
        # Maximize sum of squared bin counts (Maximize peakiness)
        # We use fixed bin number relative to range to avoid "step" artifacts
        rng = p.max() - p.min()
        bins = int(rng * 2) # 0.5 px bins
        counts, _ = np.histogram(p, bins=max(10, bins))
        return -np.sum(counts**2)

    res = minimize_scalar(
        objective, 
        bounds=(ref_angle - 10, ref_angle + 10), 
        method='bounded'
    )
    return res.x

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Assuming 'centers' is defined in your environment
    # result = analyze_grid_centers(centers, optimize_axis='x')
    # print(result)
    pass
