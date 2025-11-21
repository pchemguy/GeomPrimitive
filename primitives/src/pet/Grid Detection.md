# Grid Detection

Present grid detector, pet_geom.detect_grid_segments uses to detect line segments. Important, as of this writing, precision, width, and nfa are not used. But these metadata is probably essential for classifying major vs. minor (possibly, photometric_weight = `length * width * precision`).

## Line Segment Detection (LSD) in PET

### Overview


The PET geometry pipeline uses OpenCV's implementation of the Line Segment Detector (LSD) as the primary primitive extractor for gridlines on graph paper.

The implementation is based on:

    R. Grompone von Gioi, J. Jakubowicz, J.-M. Morel, G. Randall,
    "LSD: A Fast Line Segment Detector", IPOL / ECCV.

LSD is a fast, subpixel-accurate detector that finds straight line segments without using a Hough transform. It operates directly on the image gradients and validates each segment using an a-contrario statistical model.

### OpenCV API

PET uses a small compatibility wrapper around the OpenCV constructor:

```
def _create_lsd():
    try:
        return cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD)
    except TypeError:
        try:
            return cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except TypeError:
            return cv2.createLineSegmentDetector()
```

The returned object is an instance of OpenCV's LineSegmentDetector. The main method we use is:

```
lines, widths, precisions, nfa = lsd.detect(gray)
```

where:

gray        : single-channel uint8 or float32 image
lines       : (N, 1, 4) array of [x1, y1, x2, y2] segment endpoints
              in *image* coordinates (top-left origin, y down, float32).
widths      : (N, 1) array of estimated line widths (in pixels).
precisions  : (N, 1) array of segment precisions (higher = cleaner).
nfa         : (N, 1) array of "number of false alarms" scores
              from the a-contrario validation (lower = more significant).

In PET we immediately reshape these into flat arrays:

lines      -> (N, 4)
widths     -> (N,)
precisions -> (N,)
nfa        -> (N,)

and store them in a dictionary:

    {
        "lines":      (N, 4) float32,
        "widths":     (N,),
        "precisions": (N,),
        "nfa":        (N,),
    }

## Interpretation of LSD outputs

- lines:
    Subpixel-accurate endpoints of each detected segment. PET treats these as
    the primary geometric primitives for all subsequent angle, clustering and
    vanishing-point estimation steps.
- widths:
    An estimate of the local line half-width (or effective support width) in
    pixels, derived from the gradient structure across the line. This is a
    *direct* image-derived measurement: thicker, darker gridlines tend to
    produce larger widths. It is NOT a purely geometric length, and NOT a
    simple count-based metric.

    In PET, widths are used (together with segment length and precision) to
    construct a photometric weight per segment when distinguishing "major"
    vs "minor" grid lines.
- precisions:
    A measure of how well the local gradient orientations support the fitted
    line model for that segment. High precision means the gradients are
    tightly aligned with the segment direction (high contrast, sharp line).
    Low precision indicates noisy, blurry, or poorly aligned support.

    In PET, precision is used as a multiplicative factor in the segment
    weight (e.g. weight ~ length * width * precision) and can be used as a
    quality filter (discard very low-precision fragments).

- nfa (Number of False Alarms):
    A statistical significance score from the a-contrario model. Roughly,
    NFA is the *expected* number of times a segment at least as "good" as the
    current one would appear in random noise under the null hypothesis H0.

    Smaller NFA => more statistically significant segment.
    In the original LSD paper, segments are typically accepted if NFA < 1.

    In PET we currently keep all segments and may use nfa as a secondary
    filter if needed (e.g. discarding very large nfa outliers that look like
    noise or reflections).

## Refinement modes

`cv2.createLineSegmentDetector` accepts a "refine" flag controlling how much
post-processing is done on the raw region-based segments:

cv2.LSD_REFINE_NONE (0)
    No refinement. Raw endpoints from region growing. Fastest, lowest
    geometric accuracy. Not recommended for PET.

cv2.LSD_REFINE_STD (1)  [used by PET]
    Standard refinement: re-estimate orientation, adjust endpoints
    along the principal direction, compute width / precision / NFA
    under the full LSD model. This is typically the best trade-off
    between speed and accuracy.

cv2.LSD_REFINE_ADV (2)
    Advanced refinement; performs extra local optimization of segment
    placement. Slower and usually unnecessary for clean graph-paper
    images, but may help on extremely low-contrast input.

## Recommended usage in PET

1. Preprocessing:
    - Convert the input BGR image to grayscale (cv2.cvtColor).
    - Optionally apply mild denoising or local contrast normalization
     (e.g. CLAHE or normalized local contrast) to make segments cleaner.
2. Detection:
    - Use `_create_lsd()` to build the detector with LSD_REFINE_STD.
    - Call `lsd.detect(gray)` and convert outputs into flat arrays.
    - Do NOT threshold or normalize the image before LSD in arbitrary ways;
     LSD expects approximately linear gradients.
3. Post-processing:
    - Compute angle and length per segment.
    - Use angle-based clustering (circular statistics, KDE) to split segments
     into two principal orientation families (grid x- and y-directions).
    - Use segment *length* as the primary geometric weight and LSD `width`
     and `precision` as photometric modifiers when aggregating per-gridline
     properties (e.g. to distinguish major vs minor lines).
    - Use NFA only as an optional quality filter. Keep in mind that NFA is
     logarithmic in the original theory and is highly sensitive to noise
     and contrast.

## Numerical notes

- All LSD outputs are in image pixel coordinates (top-left origin).
- lines are float32, but subsequent PET computations typically promote to
  float64 for numerical robustness (e.g. vanishing point least-squares).
- Very short segments (length < ~2 px) are often unstable; PET may discard
  them during filtering even if LSD accepted them.

---

# ❗ What you should see if LSD is working correctly

### Width:

* values between 1px and ~4px (depending on thickness of synthetic lines)

### Precision:

* NOT constant
* values vary (e.g. 0.15, 0.22, 0.44, …)

### NFA:

* low values (e.g. 1e−3, 1e−5)
* NOT all zeros

If you see:

```
precision = 0.125 everywhere
nfa = 0 everywhere
```

→ **Your OpenCV installation is broken**
→ Install the correct wheel:

```
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

---

# ⭐ EDLines — Should you use it?

EDLines = **Edge Drawing + Line Segment Detection**
(Thanassis et al., 2014)

## Advantages over LSD

* Much **better for long lines**
* More **stable endpoints**
* Handles **broken / fragmented gridlines** better
* Fewer false positives in noisy backgrounds
* Runs faster for large images
* More recent algorithm than LSD

## Disadvantages

* No NFA or statistical validation
* No width / precision estimates
* Requires manual thresholds (though defaults are good)

## Is LSD still the top choice?

For your application (graph paper):

### LSD strengths:

* subpixel accuracy
* precise thickness estimate (**width**) ← **you need this**
* statistical validation (NFA)
* robust to non-uniform lighting
* excellent for nasty hand-held camera distortions

### EDLines strengths:

* much better at recovering **complete gridlines**
* easier merging of fragments
* faster
* more stable on long straight edges

---

# ⭐ Recommendation for PET Pipeline

Use **both**:

### 1. LSD

→ for **per-segment features** needed for classification
(line width, precision, NFA, angle dispersion)

### 2. EDLines

→ for **grid topology**, spacing, and vanishing points
(superior long-stroke stability)

This combo is standard in forensic imaging and industrial grid detection.

---
