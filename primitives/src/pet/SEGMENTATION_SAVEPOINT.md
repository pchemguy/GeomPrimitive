# Pipeline: Robust Organ Segmentation & SxL Anomaly Detection

## Overview

This pipeline implements a multi-stage computer vision workflow designed to robustly extract biological tissue (organs) from clinical images. It moves beyond simple color thresholding by integrating geometric constraints, iterative graph-cut refinement (GrabCut), and statistical percentile analysis to identify anomalies within the segmented tissue.

## The Workflow

1. **Initial Localization (Lab-a):** The image is converted to the CIELAB color space. The 'a' channel (Green-Red axis) is isolated and thresholded using Otsu's method. This provides a robust initial guess for "reddish" biological matter, ignoring lighting variations.
2. **Geometric Refinement (GrabCut):** The initial mask is treated as a "Probable Foreground." An eroded core of the mask is marked as "Definite Foreground." The GrabCut algorithm (Gaussian Mixture Models) iterates to snap the boundaries to the strongest color gradients, removing soft shadows and background noise.
3. **Morphological Standardization:** A sequence of Morphological Closing (to fill internal pinholes) and Opening (to smooth jagged edges) ensures a solid, topological region.
4. **Minimal Bounding Box:** The pipeline calculates the Minimal Area Rectangle (Rotated Bounding Box) rather than an axis-aligned box. This provides an orientation-agnostic metric for the organ's true dimensions.

## SxL Mask: Percentile Intersection Analysis

Once the **Final Dominant Mask** is generated, the pipeline performs a secondary statistical pass to identify "low-confidence" or "artifact" pixels within the organ tissue.

This is achieved via the **SxL Mask**, which targets the intersection of specific **Saturation (HSV)** and **Lightness (Lab)** percentiles relative to the object's own distribution.

### Current Threshold Configuration

The mask isolates pixels meeting the following condition:

$$Mask_{SxL} = (S < P_{25}) \land [(L < P_{10}) \lor (L > P_{85})]$$

- **Saturation ($S < 25^{th}\%ile$):** Targets the "weakest" colors in the object.
- **Lightness ($L < 10^{th}\%ile \cup L > 85^{th}\%ile$):** Targets the most extreme shadows and highlights.

### Rationale & Targeted Artifacts

The SxL mask is specifically tuned to isolate three distinct types of artifacts that deviate from the "healthy/dense" tissue baseline:

1. **Glare Points (Specular Highlights):** Characterized by **Low S + High L**. These are white reflections where the camera flash or ambient light reflects off wet tissue.
2. **Low Information / Necrosis:** Characterized by **Low S + Low L**. Dark, desaturated spots that often indicate shadows, deep cavities, or necrotic tissue.
3. **Blood Splashes:** Characterized by **Low S + Moderate-to-High L**. Unlike the dense organ tissue (which is a deep, saturated red), blood splashes often appear "thinner" (lower saturation) and shinier/wet (higher lightness).

**Distribution Note:** The presence of these artifacts (Glares + Blood Splashes) typically results in a **bimodal distribution** in the S and L histograms of the selected region. The main mode represents the organ, while the secondary modes (tails) represent these artifacts.

---

## Future Development

### TODO: Statistical Separation & Inpainting

Consider performing statistical analysis, identifying and separating the two distributions (Organ vs. Artifacts) to set thresholds dynamically rather than using fixed percentiles.

_Current Status:_ The manual threshold settings in `pet_segmentation_composite3.py` ($S_{p25}, L_{p10}, L_{p85}$) show close to optimal separation, where the thresholds on low S and high L roughly match the natural separation valleys of the bimodal distribution.

**Roadmap:**

1. **Morphological Enhancement:** The resulting SxL mask requires morphological cleaning to remove single-pixel noise.
2. **Cluster Identification:** We must differentiate between artifacts on the edge (which might just be segmentation errors) and "True Inner Clusters" (glare on the organ surface, vs. blood areas to be rejected).

Algorithm for Inner Cluster Classification & Inpainting:

To classify an inner cluster as a candidate for inpainting (e.g., glare removal), implement the following logic:

- **Size Analysis:** Calculate the cluster diameter, defined as the $95^{th}$ percentile distance between any two points within the cluster.
- **Border Proximity:** Identify the minimum Euclidean distance from the cluster's geometric center to the mask's outer border.
    - _Condition:_ Distance > $2 \times$ Cluster Size. (Ensures the spot is truly "internal" and not an edge erosion artifact).
- **Local Contrast Analysis:**
    - Identify the Minimum Area Bounding Box for the cluster.
    - **Masked Statistics:** Calculate $\mu_1$ (mean) and $\sigma_1$ (std dev) for $S$ and $L$ channels using only the masked pixels.
    - **Surround Statistics:** Double the bounding box size. Calculate $\mu_2$ and $\sigma_2$ for the _non-masked_ pixels within this expanded box (the immediate healthy neighborhood).
    - **Inpainting Trigger:** If $|\mu_2 - \mu_1| > 2 \times (\sigma_1 + \sigma_2)$, mark the cluster for inpainting.      
