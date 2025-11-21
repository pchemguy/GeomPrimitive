"""
pet_lsd_width_analysis.py
--------------------------

Ref: https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942


lsd_width_analysis.py
---------------------

High-level statistical analysis and clustering of LSD-estimated line widths.

This module implements:

    1. analyze_lsd_widths()
       ----------------------------------
       A diagnostic tool that fits Gaussian Mixture Models (GMMs) to the
       1-D distribution of per-segment line widths produced by the LSD
       (Line Segment Detector). It evaluates 1..K mixture components,
       computes AIC/BIC, selects the best-fitting model, and provides full
       diagnostics:
           - histogram + mixture PDF overlays
           - color-coded cluster histogram
           - AIC/BIC curves
           - Q-Q plots per component
           - approximate EM log-likelihood path

       The function returns a complete statistical description of the line
       width distribution:
           - sorted component means, std-devs, weights
           - per-segment responsibilities (probabilities)
           - cluster labels (or -1 for invalid inputs)
           - full GMM model object

       This routine does **not** modify or cluster the LSD structure by
       itself; it only performs the statistical analysis.

    2. cluster_line_thickness()
       ----------------------------------
       A robust 3-way classification of grid lines by apparent thickness:

            - minor  (thinner grid lines)
            - major  (thicker grid lines)
            - outliers (segments inconsistent with either cluster)

       The function consumes the output from analyze_lsd_widths() and applies
       the following rules:

           - GMM components are sorted by mean width.
             Component with smallest mean     -> minor
             Component with largest mean      -> major
             Any middle components (if K>2)   -> outlier candidates

           - Within the minor and major components, segments are accepted as
             inliers only if:
                     |width - mean| <= sigma_factor * std_dev

             where sigma_factor is typically 2.5-3.0 (robust sigma threshold).

           - All remaining segments are assigned to the outlier group.

       The result is returned **in full LSD dictionary format**, so all PET
       downstream routines can treat each group as a standalone LSD dataset:

             {
                 "minor":     LSD-dict,
                 "major":     LSD-dict,
                 "outliers":  LSD-dict,
                 "labels":    (N,) array with values {0,1,2,-1},
                 "probs":     (N,K) responsibilities,
                 "analysis":  output of analyze_lsd_widths()
             }

---------------------------------------------
Rationale
---------------------------------------------

Graph paper typically has **two distinct physical line thicknesses**:

    - a primary grid (major) with thicker lines  
    - a subgrid (minor) with thinner lines

These differences are preserved reliably even under:
    - mild blur
    - low dynamic range
    - uneven lighting
    - slight imaging noise

For reliable metric rectification and grid periodicity analysis, it is
often necessary to **separate major and minor lines**, especially when:

    - using widths as weights in clustering of line positions
    - extracting major grid spacing
    - removing subgrid noise before vanishing-point estimation
    - building robust centerline clusters
    - filtering spurious detections

Simple threshold-based approaches are insufficient because:
    - distributions are often overlapping or multimodal,
    - sample contamination exists (plastic glare, object occlusions),
    - some major lines get fragmented,
    - minor lines can become thickened by blur or reflections.

Therefore, a **GMM-based model selection (AIC/BIC)** is the most robust
approach to recovering the true bi-modal structure.

---------------------------------------------
Input / Output Format
---------------------------------------------

Both routines operate on a unified LSD descriptor structure:

    lsd = {
        "lines":      (N,4) float32   - segments [x1,y1,x2,y2]
        "widths":     (N,)  float32   - estimated LSD widths
        "precisions": (N,)  float32
        "nfa":        (N,)  float32
        "lengths":    (N,)  float32
        "centers":    (N,2) float32   - geometric midpoints
    }

analyze_lsd_widths(lsd, plot=False)
    -> returns a full statistical model (GMM) and cluster diagnostics.

cluster_line_thickness(lsd, analysis)
    -> returns:
        {
            "minor": LSD-dict,
            "major": LSD-dict,
            "outliers": LSD-dict,
            "labels":     (N,),
            "probs":      (N,K),
            "analysis":   analysis
        }

---------------------------------------------
Typical Usage in PET Pipeline
---------------------------------------------

    raw_lsd = detect_grid_segments_full(img)

    # 1. Statistical analysis
    ana = analyze_lsd_widths(raw_lsd, max_components=3, plot=True)

    # 2. Robust thickness clustering
    thick = cluster_line_thickness(raw_lsd, analysis=ana)

    lsd_minor    = thick["minor"]
    lsd_major    = thick["major"]
    lsd_outliers = thick["outliers"]

Now each group can be processed independently by:
    - gridline clustering
    - periodicity detection
    - spacing estimation
    - geometric rectification
    - outlier suppression

---------------------------------------------
Notes
---------------------------------------------

- GMM is fit only on widths that are finite and valid.
- Components are always sorted by mean width (thin -> thick).
- Outliers include:
      - middle GMM components (if K > 2)
      - non-inliers in the thin (minor) or thick (major) components
- All clustering is 1-D (width-only) and therefore extremely fast.
- The analysis function is suitable for standalone diagnostics and
  interactive tuning.

"""

from __future__ import annotations

__all__ = [
    "analyze_lsd_widths", 
    "cluster_line_thickness",
    "merge_lsd_dicts",
    "split_widths_hist",
]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Any, Optional, Tuple, List

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy import stats


def analyze_lsd_widths(
    lsd_output: Dict[str, np.ndarray],
    max_components: int = 3,
    plot: bool = False,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Analyze LSD line-width distribution and fit 1D Gaussian Mixture Models.

    This is *diagnostic*: it does not split segments, only analyzes widths.

    Args
    ----
    lsd_output:
        LSD dictionary with at least the "widths" array.
    max_components:
        Max number of GMM components (k) to test with AIC/BIC (1..k).
    plot:
        If True, produce Matplotlib diagnostics:
            - Width histogram + GMM PDFs
            - Histogram with cluster coloring (for best-k model)
            - AIC/BIC vs component count
            - Q-Q plots per cluster (best-k model)
            - Approximate log-likelihood path for best-k model
    random_state:
        Seed for GMM reproducibility.

    Returns
    -------
    dict with keys:
        "widths"           : original widths (N,)
        "mask_clean"       : boolean mask of finite widths used for fitting
        "widths_clean"     : cleaned widths (M,)
        "gmm"              : best GaussianMixture model (k chosen by BIC)
        "n_components"     : chosen k
        "labels"           : cluster labels for *all* N widths
                             (invalid / NaN -> -1)
        "probs"            : responsibilities (N, k) for valid widths,
                             NaN rows filled with 0 for invalid widths.
        "means"            : (k,) cluster means, sorted ascending
        "stds"             : (k,) cluster std-devs, sorted
        "weights"          : (k,) mixture weights, sorted to match means
        "order"            : index mapping from sorted components to
                             original GMM component indices
        "aic"              : list of AIC values for k=1..max_components
        "bic"              : list of BIC values for k=1..max_components
    """
    widths = np.asarray(lsd_output.get("widths", []), float)
    if widths.size == 0:
        raise ValueError("analyze_lsd_widths: LSD output has no 'widths' data.")

    # ------------------------------------------------------------
    # Clean widths used for fitting (finite values only)
    # ------------------------------------------------------------
    mask_clean = np.isfinite(widths)
    w_clean = widths[mask_clean]
    if w_clean.size == 0:
        raise ValueError("analyze_lsd_widths: all widths are NaN/inf.")

    # ------------------------------------------------------------
    # Fit GMM for k = 1..max_components, pick best BIC
    # ------------------------------------------------------------
    X = w_clean.reshape(-1, 1)
    aic_list: List[float] = []
    bic_list: List[float] = []
    gmms: List[GaussianMixture] = []

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
        )
        gmm.fit(X)
        gmms.append(gmm)
        aic_list.append(gmm.aic(X))
        bic_list.append(gmm.bic(X))

    # Choose best k by BIC (you can switch to AIC if you prefer)
    k_values = np.arange(1, max_components + 1)
    best_idx = int(np.argmin(bic_list))
    best_k = int(k_values[best_idx])
    best_gmm = gmms[best_idx]

    # ------------------------------------------------------------
    # Get responsibilities, means, std-devs for best model
    # ------------------------------------------------------------
    probs_clean = best_gmm.predict_proba(X)          # shape (M, k)
    labels_clean = np.argmax(probs_clean, axis=1)    # shape (M,)

    means = best_gmm.means_.ravel()
    variances = best_gmm.covariances_.reshape(-1)    # 1D because 1D GMM
    stds = np.sqrt(variances)
    weights = best_gmm.weights_.ravel()

    # Sort components by mean width (ascending: thinner -> thicker)
    order = np.argsort(means)
    means_sorted = means[order]
    stds_sorted = stds[order]
    weights_sorted = weights[order]

    # Remap labels/probs to sorted order
    # old component j -> new index idx where order[idx] == j
    inv_order = np.zeros_like(order)
    inv_order[order] = np.arange(order.size)

    labels_sorted_clean = inv_order[labels_clean]
    probs_sorted_clean = probs_clean[:, order]

    # Map labels/probs back to full N (invalid = -1, probs=0)
    labels_full = np.full(widths.shape, -1, dtype=int)
    probs_full = np.zeros((widths.size, best_k), dtype=float)
    labels_full[mask_clean] = labels_sorted_clean
    probs_full[mask_clean, :] = probs_sorted_clean

    result = {
        "widths": widths,
        "mask_clean": mask_clean,
        "widths_clean": w_clean,
        "gmm": best_gmm,
        "n_components": best_k,
        "labels": labels_full,
        "probs": probs_full,
        "means": means_sorted,
        "stds": stds_sorted,
        "weights": weights_sorted,
        "order": order,
        "aic": np.array(aic_list),
        "bic": np.array(bic_list),
        "k_values": k_values,
    }

    if not plot:
        return result

    # ============================================================
    # PLOTTING (diagnostics)
    # ============================================================

    # 1) Width histogram + GMM PDFs overlay
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 3))
    counts, bins, _ = ax1.hist(
        w_clean, bins=60, density=True, alpha=0.5, color="#1f77b4"
    )
    x_plot = np.linspace(bins[0], bins[-1], 400).reshape(-1, 1)
    # total PDF
    total_pdf = np.zeros_like(x_plot[:, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for j in range(best_k):
        mean_j = means_sorted[j]
        std_j = stds_sorted[j]
        weight_j = weights_sorted[j]
        pdf_j = weight_j * stats.norm.pdf(x_plot[:, 0], loc=mean_j, scale=std_j)
        total_pdf += pdf_j
        ax1.plot(x_plot[:, 0], pdf_j, color=colors[j], lw=1.5,
                 label=f"Comp {j}: mu={mean_j:.2f}, sigma={std_j:.2f}")
    ax1.plot(x_plot[:, 0], total_pdf, "k-", lw=2, label="Mixture PDF")
    ax1.set_title("Width Histogram + GMM PDFs")
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Density")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend()

    # 2) Histogram with cluster coloring (best-k)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3))
    for j in range(best_k):
        mask_j = labels_sorted_clean == j
        ax2.hist(
            w_clean[mask_j],
            bins=bins,
            alpha=0.6,
            color=colors[j],
            label=f"Cluster {j}",
        )
    ax2.set_title("Width Histogram (color-coded by cluster)")
    ax2.set_xlabel("Width (px)")
    ax2.set_ylabel("Count")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend()

    # 3) AIC / BIC vs components
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 3))
    ax3.plot(k_values, aic_list, "o-", label="AIC")
    ax3.plot(k_values, bic_list, "s-", label="BIC")
    ax3.axvline(best_k, color="k", ls="--", alpha=0.6,
                label=f"chosen k={best_k}")
    ax3.set_xlabel("Number of components k")
    ax3.set_ylabel("Criterion value")
    ax3.set_title("AIC / BIC vs k")
    ax3.grid(True, ls=":", alpha=0.4)
    ax3.legend()

    # 4) Q-Q plots per cluster (best-k)
    fig4, axes4 = plt.subplots(1, best_k, figsize=(4 * best_k, 4))
    if best_k == 1:
        axes4 = [axes4]
    for j in range(best_k):
        ax = axes4[j]
        data_j = w_clean[labels_sorted_clean == j]
        if data_j.size > 0:
            stats.probplot(data_j, dist="norm", plot=ax)
        ax.set_title(f"Cluster {j} Q-Q")
        ax.grid(True, ls=":", alpha=0.4)

    # 5) Approx log-likelihood path for best-k (warm-start EM)
    fig5, ax5 = plt.subplots(1, 1, figsize=(6, 3))
    max_iter_path = 20
    ll_path = []
    gmm_path = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        warm_start=True,
        max_iter=1,
        random_state=random_state,
    )
    # Initialise once
    gmm_path.fit(X)
    ll_path.append(gmm_path.score(X) * X.shape[0])
    for _ in range(max_iter_path - 1):
        gmm_path.max_iter += 1
        gmm_path.fit(X)
        ll_path.append(gmm_path.score(X) * X.shape[0])

    ax5.plot(np.arange(1, max_iter_path + 1), ll_path, "o-")
    ax5.set_xlabel("Iteration (approx)")
    ax5.set_ylabel("Log-likelihood")
    ax5.set_title("Approx. log-likelihood path (best-k model)")
    ax5.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    plt.show()

    return result


def _slice_lsd(lsd: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Helper: apply boolean mask to all LSD arrays."""
    return {
        "lines":      lsd["lines"][mask],
        "widths":     lsd["widths"][mask],
        "precisions": lsd["precisions"][mask],
        "nfa":        lsd["nfa"][mask],
        "lengths":    lsd["lengths"][mask],
        "centers":    lsd["centers"][mask],
    }


def cluster_line_thickness(
    lsd_output: Dict[str, np.ndarray],
    analysis: Optional[Dict[str, Any]] = None,
    robust_sigma: float = 3.0,
) -> Dict[str, Any]:
    """
    Split LSD segments into four groups based on line *width*:

        - "minor"      : thinner grid lines
        - "major"      : thicker grid lines
        - "outlier_lo" : too thin
        - "outlier_hi" : too thick


    Strategy (option B):
        1) Run analyze_lsd_widths() (or use an existing analysis) to get
           a 1D GMM over widths; components are sorted by mean width.
        2) Assign each segment to the most probable component.
        3) Label the *smallest-mean* component as "minor",
           largest-mean component as "major". Intermediate ones (if any)
           are treated as potential outlier clusters.
        4) Within minor/major components, mark any width that lies
           further than robust_sigma standard deviations from its
           component mean as "outlier".

    Args
    ----
    lsd:
        LSD dictionary with lines + widths + metadata.
    analysis:
        Optional precomputed result from analyze_lsd_widths().
        If None, analyze_lsd_widths(lsd, max_components=3)
        is called internally with plot=False.
    robust_sigma:
        Number of sigma around each component mean used for inlier/outlier
        decision. Default ~3sigma.

    Returns
    -------
    dict with:
        "minor"       : LSD dict (same keys as input) for minor lines
        "major"       : LSD dict for major lines
        "outliers_lo" : LSD dict for thinner outliers
        "outliers_hi" : LSD dict for thicker outliers
        "labels"      : full label array of length N:
                           0 = minor, 1 = major,
                           2 = outliers_lo, 3 = outliers_hi,
                          -1 = invalid
        "probs"       : responsibilities from GMM (N, k)
        "analysis"    : the analysis dict from analyze_lsd_widths()
    """
    # Run or reuse analysis
    if analysis is None:
        analysis = analyze_lsd_widths(lsd_output, max_components=3, plot=False)

    widths = np.asarray(analysis["widths"], float)
    labels_gmm = np.asarray(analysis["labels"], int)  # 0..k-1 or -1
    probs = np.asarray(analysis["probs"], float)      # (N, k)
    means = np.asarray(analysis["means"], float)      # sorted ascending
    stds = np.asarray(analysis["stds"], float)
    weights = np.asarray(analysis["weights"], float)  # weights matching sorted means
    k = int(analysis["n_components"])

    if k < 2:
        raise RuntimeError(
            "cluster_line_thickness: best GMM has k<2 components; "
            "cannot form minor/major thickness groups."
        )

    N = widths.size
    if labels_gmm.shape[0] != N:
        raise ValueError("cluster_line_thickness: labels size mismatch with widths.")

    # --------------------------------------------------------
    # INTELLIGENT COMPONENT SELECTION
    # --------------------------------------------------------
    # Instead of blindly taking 0 and k-1, we take the two components
    # with the highest mixture weights (the two dominant peaks).
    
    # 1. Find indices of the two largest weights
    # argsort is ascending, so take the last two
    dominant_indices = np.argsort(weights)[-2:]
    
    # 2. Sort those indices back by mean (idx 0 is thinner than idx 1)
    # This ensures idx_minor points to the thinner of the two dominant peaks
    idx_minor, idx_major = sorted(dominant_indices)

    minor_comp = idx_minor
    major_comp = idx_major
    
    # All other components are treated as outlier clusters
    middle_comps = set(range(k)) - {minor_comp, major_comp}

    labels_final = np.full(N, -1, dtype=int)

    # Mask of valid GMM assignments
    valid_mask = labels_gmm >= 0

    # Pull width, component, mean, std for each valid sample
    comp_idx = labels_gmm.copy()  # 0..k-1 or -1
    w_valid = widths[valid_mask]
    c_valid = comp_idx[valid_mask]

    # Inlier masks (within robust_sigma * std)
    minor_mean = means[minor_comp]
    minor_std = stds[minor_comp]
    major_mean = means[major_comp]
    major_std = stds[major_comp]

    inlier_minor = np.zeros_like(w_valid, dtype=bool)
    inlier_major = np.zeros_like(w_valid, dtype=bool)

    # for minor component
    mask_minor = c_valid == minor_comp
    inlier_minor[mask_minor] = (
        np.abs(w_valid[mask_minor] - minor_mean) <= robust_sigma * max(minor_std, 1e-6)
    )

    # for major component
    mask_major = c_valid == major_comp
    inlier_major[mask_major] = (
        np.abs(w_valid[mask_major] - major_mean) <= robust_sigma * max(major_std, 1e-6)
    )

    # All 'middle' (non-dominant) components AND any non-inliers -> outliers
    is_middle = np.isin(c_valid, list(middle_comps))
    is_outlier_valid = (
        (~inlier_minor) & (~inlier_major)
    ) | is_middle

    # Assign final labels for valid entries
    # 0 = minor, 1 = major, 2 = outliers
    labels_valid_final = np.full_like(c_valid, 2, dtype=int) 
    labels_valid_final[inlier_minor] = 0
    labels_valid_final[inlier_major] = 1

    # Transfer back to full-size labels
    labels_final[valid_mask] = labels_valid_final

    # Boolean masks for each group
    minor_mask = labels_final == 0
    major_mask = labels_final == 1
    # Note: We group all outliers (statistical + non-dominant clusters) together here, 
    # but you can split them if needed.
    outlier_mask = labels_final == 2
    # Slice LSD structures

    minor_lsd = _slice_lsd(lsd_output, minor_mask)
    major_lsd = _slice_lsd(lsd_output, major_mask)
    outlier_lsd = _slice_lsd(lsd_output, outlier_mask)

    # --------------------------------------------------------
    # SPLIT OUTLIERS & PREPARE OUTPUT 
    # --------------------------------------------------------
    
    # Create masks
    minor_mask = labels_final == 0
    major_mask = labels_final == 1
    outlier_mask = labels_final == 2

    # Split outliers using Major Mean as threshold
    #   outliers_lo: outliers thinner than the major peak (includes 'valley' noise)
    #   outliers_hi: outliers thicker than the major peak
    outliers_lo_mask = outlier_mask & (widths < major_mean)
    outliers_hi_mask = outlier_mask & (widths >= major_mean)

    # Update labels for specific outlier types
    # 0=minor, 1=major, 2=outlier_lo, 3=outlier_hi
    labels_final[outliers_lo_mask] = 2
    labels_final[outliers_hi_mask] = 3

    outliers_lo_lsd = _slice_lsd(lsd_output, outliers_lo_mask)
    outliers_hi_lsd = _slice_lsd(lsd_output, outliers_hi_mask)

    return {
        "minor": minor_lsd,
        "major": major_lsd,
        "outliers_lo": outliers_lo_lsd,
        "outliers_hi": outliers_hi_lsd,
        "labels": labels_final,
        "probs": probs,
        "analysis": analysis,
    }


def merge_lsd_dicts(a: dict, b: dict) -> dict:
    """
    Merge two LSD-formatted dictionaries by concatenating all fields.

    Each input must contain:
        "lines"      : (N,4)
        "widths"     : (N,)
        "precisions" : (N,)
        "nfa"        : (N,)
        "lengths"    : (N,)
        "centers"    : (N,2)

    Output:
        A NEW LSD dict with all arrays concatenated row-wise.
    """

    REQUIRED = [
        "lines", "widths", "precisions", "nfa",
        "lengths", "centers"
    ]

    # --- validate both ---
    for name in REQUIRED:
        if name not in a:
            raise KeyError(f"LSD dict A missing key '{name}'")
        if name not in b:
            raise KeyError(f"LSD dict B missing key '{name}'")

    # --- convert to arrays ---
    def arr(x): return np.asarray(x)

    A = {k: arr(a[k]) for k in REQUIRED}
    B = {k: arr(b[k]) for k in REQUIRED}

    # --- check dimensional compatibility ---
    if A["lines"].ndim != 2 or B["lines"].ndim != 2:
        raise ValueError("lines arrays must be 2D")

    if A["centers"].shape[1] != 2 or B["centers"].shape[1] != 2:
        raise ValueError("centers arrays must have shape (N,2)")

    # --- concatenate ---
    merged = {
        "lines":      np.vstack([A["lines"],      B["lines"]]).astype(np.float32),
        "widths":     np.hstack([A["widths"],     B["widths"]]).astype(np.float32),
        "precisions": np.hstack([A["precisions"], B["precisions"]]).astype(np.float32),
        "nfa":        np.hstack([A["nfa"],        B["nfa"]]).astype(np.float32),
        "lengths":    np.hstack([A["lengths"],    B["lengths"]]).astype(np.float32),
        "centers":    np.vstack([A["centers"],    B["centers"]]).astype(np.float32),
    }

    return merged


def split_widths_hist(
    clusters: dict,
    bins: int = 80,
    title: str = "LSD Width Histogram (4-Cluster Split)",
    alpha=0.60,
    class_colors=None,
):
    """
    Draw a color-coded histogram of FOUR width clusters.
    Input is a dict with four LSD sub-dicts:

        clusters = {
            "minor": {... lsd dict ...},
            "major": {... lsd dict ...},
            "outliers_lo": {... lsd dict ...},
            "outliers_hi": {... lsd dict ...},
        }

    Each lsd dict **must contain**:
        - "widths": array(N,)

    This routine DOES NOT require labels or any preprocessing.
    """
    # Default palette
    if class_colors is None:
        class_colors = {
            "minor":        "#1f77b4",  # blue
            "major":        "#d62728",  # red
            "outliers_lo": "#2ca02c",  # green
            "outliers_hi": "#9467bd",  # purple
        }

    # ----------------------------------------------------
    # 1) COLLECT ALL WIDTHS FIRST - GLOBAL RANGE
    # ----------------------------------------------------
    all_widths = []

    ordered = ["outliers_lo", "minor", "major", "outliers_hi", "outliers",]

    for name in ordered:
        if name in clusters and clusters[name] is not None:
            w = clusters[name].get("widths", None)
            if w is not None and len(w) > 0:
                all_widths.append(np.asarray(w, float))

    if len(all_widths) == 0:
        print("split_widths_hist: No width data found.")
        return

    all_widths = np.concatenate(all_widths)
    wmin, wmax = float(np.min(all_widths)), float(np.max(all_widths))

    # Establish global bin edges
    bin_edges = np.linspace(wmin, wmax, bins + 1)

    # ----------------------------------------------------
    # 2) PLOT USING ONE BIN EDGE SET
    # ----------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=16, weight="bold")

    for name in ordered:
        if not name in clusters:
            continue
        lsd = clusters[name]
        if lsd is None:
            continue

        w = lsd.get("widths", None)
        if w is None or len(w) == 0:
            continue

        # Plot using unified bin edges
        plt.hist(
            np.asarray(w, float),
            bins=bin_edges,
            alpha=alpha,
            color=class_colors.get(name, "gray"),
            label=f"{name} (n={len(w)})",
            edgecolor="none",
        )

    plt.xlabel("LSD width (pixels)")
    plt.ylabel("Count")
    plt.grid(True, linestyle=":", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()
