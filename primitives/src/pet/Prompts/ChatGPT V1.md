https://chatgpt.com/c/69247ad4-57d4-8326-be03-21c0297a48be
https://chatgpt.com/c/692481f3-0b30-832f-b5db-4d98381939e2

# **Deep Research Prompt**

## **Title:**

**State of the Art in ML-Free Analysis & Distortion Correction of Lab Photos Containing Graph-Paper Grids**

## **Prompt**

Conduct a **deep technical literature and tooling review** on **machine-learning-free methods** (classical image processing, geometry, signal processing, numerical optimization, photogrammetry) for analyzing **laboratory photographs containing millimeter graph paper**.

The goal is to identify **state-of-the-art algorithms**, **workflows**, and **robust engineering strategies** for a **fully automated pipeline** that can:

1. **Detect and extract the grid** (millimeter, 1 mm or 5 mm).    
2. **Estimate and analyze distortions**:
    - projective distortion (rotation, perspective)
    - affine skew
    - radial distortion (barrel/pincushion)
    - uneven scaling
    - rolling-shutter-like warps (if relevant)
3. **Correct distortions** to recover:
    - orthogonality
    - uniform spacing
    - metric scale (absolute mm calibration)
4. **Estimate spacing** (grid spacing in px/mm, per-axis)
5. **Perform measurement of samples**:
    - linear size
    - area (polygon, irregular regions)
6. **Operate reliably under adverse conditions**, including:
    - suboptimal & uneven lighting
    - shadows and specular glares (plastic sleeves, wet samples, glass)
    - partial occlusion of the grid by samples
    - partial grid visibility (e.g., only a corner visible)
    - non-uniform backgrounds
    - slight blur / out-of-focus
    - chromatic aberration
    - JPEG compression artifacts
    - moderate lens distortion from phone cameras

---

## **Research Requirements**

### **1. Core algorithm families**

Identify and deeply analyze classical (non-ML) algorithm families relevant to this pipeline:

- **Edge-based grid detection:**
    - Sobel/Canny, Hough + clustering strategies
    - LSD (Line Segment Detector)
    - EDLines, ELSED, MLSD-free alternatives
- **Grid topology reconstruction:**
    - intersection clustering
    - K-means on line angles
    - frequency-domain detection (FFT peaks for grid-like patterns)
    - Radon and Hough space periodicity detection
- **Projective/perspective correction:**
    - homography estimation from vanishing points & line families
    - two- or three-vanishing-point solutions
- **Camera calibration–free radial distortion correction:** 
    - plumb-line methods
    - straight-line optimization
    - polynomial & division models (1-parameter κ, 2-parameter models)
    - using grid rectification as a constraint
    - self-calibration approaches (Fitzgibbon, Devernay)
- **Robust sampling / RANSAC-based geometry:**
    - line fitting
    - cluster rejection
    - outlier removal
- **Local & global spacing estimation:**
    - regularity constraints
    - lattice estimation
    - subpixel refinement
    - autoregressive periodicity models
- **Pixel-to-mm calibration without ML:**
    - direct grid spacing inference
    - multi-grid consistency scoring
    - local vs global scaling
- **Robust lighting correction:**
    - CLAHE
    - Retinex (multiscale & variants)
    - homomorphic filtering
    - shading field estimation
    - highlight removal

### **2. Pipeline architectures**

Seek **complete workflows**, even if they appear in scattered components.

Emphasize:
- Fully automatic pipelines (no manual clicks)
- Fail-fast or progressive refinement strategies
- Multi-stage pipelines:
    1. Preprocessing →
    2. Grid detection →
    3. Line clustering →
    4. Vanishing point estimation →
    5. Homography →
    6. Radial distortion →
    7. Metric calibration →
    8. Post-warp validation →
    9. Measurement extraction

Include references for well-designed end-to-end pipelines used in:
- Forensics
- Document analysis
- Architectural drawing rectification
- Aerial-photo orthorectification
- Planar pattern photogrammetry

### **3. Corner cases & robustness strategies**

Focus specifically on methods robust to:
- **Specular glare**
    - polarization-invariant transforms
    - dark-channel prior
    - highlight thresholding + inpainting
- **Occlusions**
    - grid continuation using periodicity priors
    - missing-line interpolation
- **Non-uniform illumination**
    - shading model estimation
    - low-frequency lighting correction
- **Partial grid availability**
    - statistical recovery of lattice parameters
- **Non-square pixels / anamorphic distortion**
- **Strong JPEG blocking** (grid-like patterns confuse line detectors)
- **Color-channel inconsistencies**
    - using L* channel for structure extraction

### **4. Validation and quality metrics**

Identify evaluation metrics for:
- Grid orthogonality
- Spacing uniformity
- Radial distortion residual
- Homography correctness
- Robustness to noise, blur, JPEG artifacts

Include classical metrics and engineering heuristics (e.g., consistency of spacing over lattice, line straightness post-correction).

### **5. Implementation details & known good libraries**

List and evaluate open-source tools (ML-free):
- OpenCV (LSD, Hough, calibration, homographies)
- scikit-image (FFT tools, ridge detection, corner detection)
- Leptonica (document analysis)
- libCVD
- BoofCV (strong plane-model + calibration features)
- imutils
- computational geometry packages

Identify which algorithms inside these libraries are best suited, how to combine them, and typical failure modes.

### **6. Novel or lesser-known techniques**

Include:
- frequency-domain grid reconstruction
- integer-lattice estimation from noisy projections
- symmetries / autoconvolution
- helmoltz stereopsis–inspired approaches
- geometry-based self-calibration (using straight lines & orthogonality constraints)
- projective dual-space reasoning

---

## **Deliverables**

The output should be a **comprehensive technical report** organized into sections:
1. Executive Summary
2. Problem Definition
3. Required Capabilities
4. Algorithm Families (with strengths/weaknesses)
5. Pipeline Blueprints (2–4 options)
6. Robustness & Failure Mode Analysis
7. Practical Implementation Recommendations 
8. Gaps in Current Methods
9. Suggested Research Directions
10. Sources, Papers, and Open-Source Repos

The report should be **engineering-focused**, providing **algorithmic recipes**, pseudo-code, comparison tables, and practical considerations.

---

## **Additional Emphasis**

Ensure the research prioritizes:
- **ML-free approaches only.**
- **Explainability and determinism.**
- **Repeatability across large batches of lab photos.**
- **Parameter tuning strategies** for repeatable acquisition setups (same camera, same bench, similar grid).
- **Integration into an automated production pipeline** with no manual calibration.
