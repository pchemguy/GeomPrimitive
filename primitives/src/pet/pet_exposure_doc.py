"""
Tuning Instructions (Graph-Paper Exposure Pipeline)
===================================================

The PET exposure pipeline is designed to correct uneven lighting, restore 
visible detail in shadowed graph-paper regions, and normalize the overall 
appearance of lab photographs. Use the parameters below to control behavior 
and adapt the pipeline to different lighting conditions.

------------------------------------------------------
1) White Region Detection (GMM)
------------------------------------------------------

The shadow-aware GMM-based detector segments the graph paper even under 
strong uneven lighting. The pipeline emphasizes (a,b) chroma over luminance 
to keep bright and shadowed paper in the same cluster.

Tunables:
    n_components (default: 3)
        More components allow finer segmentation but may overfit noise.
        Use 3–4 for typical lab backgrounds.

    prob_threshold (default: 0.35)
        Minimum posterior probability for classifying a pixel as paper.
        Increase to 0.45 to get a stricter mask, lower to 0.25 for more 
        inclusive masks on heavily shadowed images.

    sample_fraction (default: 0.10–0.15)
        Fraction of pixels sampled for GMM fitting. Higher values improve 
        stability but increase compute time.

    highlight_clip (default: 99.8)
        Clips extreme highlight values before clustering. Lower this 
        to suppress specular reflections more aggressively.

------------------------------------------------------
2) Mask Refinement
------------------------------------------------------

The connected-component refinement stage ensures the mask is contiguous and 
covers the entire sheet of paper, avoiding bright-only masks that would break 
auto-levels.

Tunables:
    close_ksize (default: 7)
        Morphological closing kernel. Increase for denser grids or when 
        grid lines break mask cohesion.

    min_area_frac (default: 0.01)
        Minimum fraction of frame that must belong to the largest connected 
        paper region. Increase if small bright objects are mistakenly selected.

------------------------------------------------------
3) White Balance (Diagonal Gain)
------------------------------------------------------

WB uses mean BGR values in the refined mask. Gains are clamped to avoid 
over-correction and color shifts.

Tunables:
    gain_min, gain_max (default: 0.5–2.0)
        Bounds on channel gain factors. Use a narrower range if colors remain 
        unstable or a wider range when lighting is extreme.

------------------------------------------------------
4) Auto-Levels (Masked With Guardrails)
------------------------------------------------------

Auto-levels normally uses masked paper pixels to determine the tonal range, 
but will fall back to global statistics if the masked range is too small. 
This prevents crushing the dark side of the paper to black.

Tunables:
    low_clip, high_clip (default: 0.5% / 99.5%)
        Percentiles used for black/white points. For flatter lighting, reduce 
        low_clip to 0.2% and raise high_clip to 99.8%.

    min_range (default: 30)
        Minimum dynamic range in masked region before falling back to global. 
        Increase if fallback triggers too often.

    lo_max (default: 80)
        Maximum allowed black point. Raising this brightens the entire image; 
        lowering it preserves contrast but may leave deep shadows.

------------------------------------------------------
5) Local Detail Enhancement (Optional)
------------------------------------------------------

OpenCV's detailEnhance increases local contrast and improves visibility of 
the graph paper grid without altering global color balance.

Tunables:
    sigma_s (default: 12)
        Spatial smoothing (0–200). Larger values give stronger smoothing.

    sigma_r (default: 0.2)
        Range smoothing (0–1). Higher values reduce edge enhancement strength.

------------------------------------------------------
Recommended Starting Values
------------------------------------------------------

These settings work well for most lab photographs of graph paper:

    detect_white_regions_gmm:
        n_components=3
        prob_threshold=0.35
        highlight_clip=99.8

    refine_paper_mask:
        close_ksize=7
        min_area_frac=0.01

    whitebalance_auto_graphpaper:
        gain_min=0.5, gain_max=2.0

    auto_levels_masked:
        low_clip=0.005
        high_clip=0.995
        min_range=30
        lo_max=80

    exposure_detail_enhance:
        sigma_s=12
        sigma_r=0.2

------------------------------------------------------
Notes
------------------------------------------------------

* If the shadowed region still looks too dark, lower `lo_max` to 60 or increase 
  `detailEnhance` parameters.

* If the paper mask excludes part of the paper, increase `prob_threshold` or 
  reduce `close_ksize`.

* If auto-levels produces excessive brightening, reduce `high_clip` or 
  disable `detailEnhance`.

* For extremely uneven lighting, consider adding a Retinex-style pre-correction 
  before white balance.

* The pipeline is tuned for graph paper, but it generalizes well to any 
  white-background lab imagery.

"""
