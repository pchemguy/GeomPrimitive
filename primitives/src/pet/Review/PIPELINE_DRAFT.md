> [!NOTE]
> 
> I have a set of real lab pictures with millimeter graph paper in background and I want do define and develop enhancement pipeline. Assume the actual background paper has white color and the lighting is suboptimal (not enough light). I need recipes and optimal order for these tasks:
> 
> - assess the amounts of various standard noise types and apply automatic denoising
> - adust brightness, perhaps curves by setting white point to lighter regions of paper, which also needs to be identified.
> 
> Perhaps, a more interesting objective is to assess geometric distortions by looking at the grids. Can I identify grids via fft/wavelet? Can I estimate parameters for reversing rotated camera view (all three axes) and pinhole camera distortions by analyzing spatial grid info?
>
> My guess I will need opencv, skimage, what else?

---
---

Below is a **complete, engineer-level recipe** for building an enhancement + camera-geometry assessment pipeline **specifically for real lab photographs with millimeter graph paper**, where **paper is white with faint grid** and **lighting is suboptimal**.

This is written as a **design document** with reasoning, processing order, and concrete algorithms / library choices.

---

# ✅ 0. Recommended Libraries

You'll need:

### **Core**

* **OpenCV** (`cv2`) → filtering, geometry, camera models
* **scikit-image** → denoising, exposure, FFT tools
* **NumPy**, **SciPy (fft, ndimage)** → FFT & correlation
* **OpenCV contrib** (optional) → better calibration
* **pywavelets** → wavelet-based grid detection

### **Optional / advanced**

* **imreg_dft** → robust sub-pixel registration and affine detection via FFT
* **OpenCV ArUco/Charuco** if you add manual calibration markers

---

# ✅ 1. Pipeline Overview (Optimal Order)

This is the recommended full pipeline:

1. **RAW input (BGR/RAW/PNG)**
2. **White region detection → estimate “paper whiteness”**
3. **Exposure correction**
    * White-point correction based on paper
    * Gamma / curve adjustment
4. **Noise estimation (quantify each noise type)**
    * Gaussian, shot noise, salt-pepper
    * Then **auto-denoising** using best model
5. **Grid extraction & enhancement**
    * FFT or wavelets to identify grid frequencies
    * Extract grid lines (major + minor, depending on paper)
6. **Geometric distortion estimation**
    - **Perspective / camera orientation (3 DOF)**
    - **Skew / affine distortions**
    - **Lens distortion params (k₁, k₂, p₁, p₂)**
7. **Back-projective correction**
    * Remove perspective
    * Remove radial distortion
8. **Quality check & refine**

---

# 🔧 2. Lighting & Exposure Correction

## **2.1 Detect the “white paper” baseline**

Your background graph paper *is* your calibration target.

### Strategy:

1. Apply a **large Gaussian blur** → smooth out grid lines:

   ```python
   blurred = cv2.GaussianBlur(img, (51, 51), 0)
   ```
2. Compute the **top N% brightest pixels** → these approximate paper.
3. Use **95–98th percentile** of blurred grayscale as white point estimate.

Or use skimage:

```python
from skimage.filters import threshold_otsu
val = threshold_otsu(blurred_gray)
```

**Result:**
You get estimated paper white color → `(W_R, W_G, W_B)`.

### White-point correction:

Use **Gray World** or **White Patch** algorithm:

```python
scales = target_white / measured_white
img_corr = img * scales
```

### Remove uneven illumination:

Use **Retinex** (skimage works well):

```python
from skimage.restoration import denoise_bilateral
illum = cv2.GaussianBlur(gray, (101,101),0)
corrected = gray / (illum + eps)
```

---

# 🔧 3. Noise Estimation + Denoising

## **3.1 Automatically Estimate Noise**

Use:

### Gaussian noise estimate

* skimage: `estimate_sigma(image, multichannel=True)`
* wavelet-based: PyWavelets variance in HH subbands

### Shot noise (Poisson)

* Check relationship between mean and variance locally:
  If var ≈ mean → Poisson-like.

### Salt & pepper

* Measure pixel impulse proportion using
  `np.count_nonzero((img < t1) | (img > t2)) / total`.

---

## **3.2 Apply the right denoiser**

| Noise type | Best denoiser                                                     |
| ---------- | ----------------------------------------------------------------- |
| Gaussian   | **Non-local Means**, **BM3D**, CV’s `fastNlMeansDenoisingColored` |
| Poisson    | **Wavelet shrinkage**, **Anscombe transform → BM3D**              |
| Mixed      | **skimage “denoise_nl_means”** or **CV bilateral filter**         |

Recommendation:

```python
from skimage.restoration import denoise_nl_means, estimate_sigma
sigma = estimate_sigma(img, channel_axis=-1)
den = denoise_nl_means(img, h=1.2*sigma, patch_size=7, patch_distance=9)
```

---

# 🔧 4. Grid Detection (FFT/Wavelet)

Your graph paper is periodic → perfect for frequency analysis.

## **4.1 FFT method**

1. Convert to grayscale
2. FFT → magnitude spectrum
3. Detect **strong peaks** away from DC.
   There will be **4 dominant peaks** (two axes × ±).

Grid frequency `fx, fy` is given by peak locations.

### From FFT peaks you get:

* Grid spacing (mm to pixel scale)
* Rotation angle of camera (2D rotation)
* Perspective shear (peak asymmetry)
* Local warping (if peaks are broadened)

FFT is excellent for:

* Dense minor grids (1 mm)
* Major grids (5–10 mm)

## **4.2 Wavelet method**

Using 2D wavelet transforms:

* CWT ridge detection → grid lines
* Use directional wavelets (Morlet/Pauli) to detect line orientation

Wavelet method is excellent for:

* Uneven lighting
* Partial occlusion
* Low contrast grids

---

# 🎯 5. Geometric Distortion Estimation

This is the key part if you want **camera pose** and **lens distortion**.

## ✔ 5.1 Perspective + Camera Rotation

Grid lines should be **parallel**, but under perspective they converge.

### How to estimate:

* Extract grid lines via Sobel/Canny → Hough transform → detect line families
* Two line families → compute vanishing points:
    * Intersection of horizontal lines → VP₁
    * Intersection of vertical lines → VP₂
* Third vanishing point is either inferable (orthogonality assumption) or unnecessary.

### From VPs:

* Recover camera yaw, pitch, roll
* Estimate homography H mapping image → canonical grid

OpenCV:

```python
H, _ = cv2.findHomography(src_grid_points, dst_ideal_points)
```

---

## ✔ 5.2 Radial Lens Distortion (k₁, k₂, p₁, p₂)

Radial distortion makes straight grid lines **curve**.

### Approach:

1. Use extracted grid lines
2. Fit curves (splines)
3. Estimate distortion by matching curved lines to ideal straight lines.

OpenCV supports this:

```python
camera_matrix, dist_coeffs = cv2.calibrateCamera(objpoints, imgpoints, ...)
```

You can provide:

* objpoints: ideal grid (mm scale)
* imgpoints: detected intersections (via FFT/Hough)

Even without calibration pattern, grid lines from paper serve as a *pseudo-calibration target*.

---

## ✔ 5.3 Correct order:

1. Estimate & remove ISS:
    * Radial distortion first
    * Then perspective (homography)
    * Then affine corrections
2. Confirm grid lines become straight.

---

# 🔧 6. Back-Projection & Image Correction

### Step 1: Undistort radial lens

```python
undist = cv2.undistort(img, camera_matrix, dist_coeffs)
```

### Step 2: Perspective correction (rectify the paper)

```python
flat = cv2.warpPerspective(undist, H, output_size)
```

After this the grid should appear straight and equally spaced.

---

# 🔍 7. Final enhancements (optional)

* Local contrast enhancement (CLAHE)
* Sharpening only after full denoise/dewarp
* Extract object boundaries on the grid

---

# 🧩 8. Practical Notes for Lab Picture Use-Case

### **What the grid gives you:**

* Perfect periodic frequency structure → FFT friendly
* Perfect rectilinear geometry → camera calibration friendly
* Known metric spacing (1 mm, 5 mm, 10 mm) → scale calibration

### **Best feature extractor for intersections**

* Harris with subpixel refinement
* ORB corners (fast)
* Peaks in distance transform from primary grid lines

### **Avoid pitfalls**

* Uneven lighting → kills FFT → apply illumination correction first
* Over-denoising → destroys faint grid lines
  → denoise *after* exposure correction but *before* FFT
* Too strong CLAHE → will artificially strengthen noise
  → apply conservatively or after rectification

---

# 📦 9. Minimal Working Pipeline (Pseudo-Code)

```python
img = load_image()

# 1. White paper detection & brightness correction
white_estimate = estimate_paper_white(img)
img = white_balance(img, white_estimate)
img = illumination_correct(img)

# 2. Noise estimation + denoise
sigma = estimate_sigma(img)
img = denoise_nl_means(img, h=1.2*sigma)

# 3. Extract grid via FFT
fx, fy, angle = detect_grid_fft(img)
grid_lines = extract_grid_lines(img)

# 4. Compute vanishing points & perspective
Vp1, Vp2 = compute_vanishing_points(grid_lines)
H = compute_homography_from_vps(Vp1, Vp2)

# 5. Estimate distortion
camera_matrix, dist_coeffs = estimate_lens_distortion(grid_lines)

# 6. Correct geometry
img = cv2.undistort(img, camera_matrix, dist_coeffs)
img = cv2.warpPerspective(img, H)

# 7. Final enhancement
img = apply_clahe(img)
```

---

# 🔮 10. If needed, I can deliver:

* Full Python implementation with all steps
* FFT grid detector module
* Lens calibration using arbitrary grid images
* Grid intersection detection
* Path-wise warping correction for your synthetic grid generator
* Or integrate this into your existing **hand-drawn paper** pipeline

---

If you want, I can generate:

✅ **Complete ready-to-run Python module**
OR
✅ **Visualization tools to debug grid extraction**
OR
✅ **Camera parameter solver from your dataset**

Just tell me which direction to go next.
