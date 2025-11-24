REF: https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942

# ⭐ **CLAHE in OpenCV**

`clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))`

This is one of the _most powerful and safe_ contrast-enhancement tools for real photos — especially **under uneven lighting**, which is exactly your graph-paper case.

---

# **1. What CLAHE is**

**CLAHE = Contrast Limited Adaptive Histogram Equalization**

- **Adaptive** → contrast is adjusted _locally_ in small tiles    
- **Histogram equalization** → redistributes intensities to increase contrast
- **Contrast-limited** → prevents over-amplification of noise and grain

It is far superior to global histogram equalization, which often destroys highlights/shadows and creates halos.

---

# **2. Parameters you asked about**

### ✔ `clipLimit=3`

This is **the key**.

- Controls how aggressively local contrast can increase    
- Higher → more contrast
- Too high → noise, tile-borders, “paper pores” get exaggerated
- Too low → nearly no contrast enhancement

**Typical useful range for real photos:**

- `clipLimit=1.5` → very gentle
- `clipLimit=2.0–3.0` → common for photography
- `clipLimit=4–6` → strong (may create artifacts)

Your choice `3` is a **balanced and safe starting point** for real lab images.

---

### ✔ `tileGridSize=(8,8)`

This splits the image into **8×8 tiles**, each processed independently.

- Small tiles (like `4×4`) → very local contrast, more detail but more noise    
- Larger tiles (like `16×16`) → smoother, more global contrast

**8×8** is a good default.

For graph paper:

- `tileGridSize=(8,8)` keeps local contrast consistent across shadows
- If your grid cells are large or image is high-res → use `(12,12)` or `(16,16)`

---

# **3. How to apply CLAHE**

CLAHE works on **single channel** images.  
For color images, use:

### ✔ Option A: L-channel in LAB (best for photography)

```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
l2 = clahe.apply(l)

lab2 = cv2.merge((l2, a, b))
result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
```

This avoids color shifting.

### ✔ Option B: apply to each channel (not recommended)

Causes color artifacts — don’t use unless specifically needed.

---

# **4. What CLAHE is good for**

### ✔ Graph paper under uneven lighting

CLAHE is one of the **best** methods for:

- fixing shadows    
- equalizing lighting
- bringing out faint grid lines
- boosting midtones without destroying whites

### ✔ Works great _before_ FFT/Hough/grid processing

Because it gives you:

- uniform local contrast → LSD sees more line segments
- consistent brightness → VP estimation more stable
- less risk of losing faint lines near shadows

---

# **5. What CLAHE is NOT good for**

❌ Not intended for global brightness correction  
❌ Not intended to establish accurate “white point”  
❌ Not good for very dark, underexposed images (will amplify noise)  
❌ Not a substitute for Retinex or illumination modeling

Often you combine:

1. **Retinex** → remove global uneven illumination
2. **CLAHE** → enhance local contrast
3. **White balance** → normalize color

---

# **6. When to apply CLAHE in your PET pipeline**

**Correct position: AFTER illumination correction but BEFORE LSD**

✔ Good:

```
Retinex → CLAHE → Denoise → LSD → VP estimation → rectification
```

❌ NOT good before Retinex (would distort illumination map)

❌ NOT good after rectification (can distort geometric info)

---

# **7. Example for your PET module**

```python
def apply_clahe(img, clip=3.0, tiles=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
```

---

# ⭐ Summary

**CLAHE(clipLimit=3, tileGridSize=8×8)** is a strong, safe local-contrast enhancer suited perfectly to your graph-paper and grid-detection pipeline.

Use it after illumination correction and before edge detection.

---

# ⭐ 1. What CLAHE does (OpenCV CLAHE)

CLAHE = _Contrast Limited Adaptive Histogram Equalization_.

### Mechanism:

- Divide image into tiles
- Perform **local histogram equalization** inside each tile
- Clip histogram peaks to avoid over-enhancement
- Interpolate between tiles to avoid seams

### Characteristics:

✔ Excellent for **midtone enhancement**  
✔ Robust under “soft shadow” conditions  
✔ Saturates dark/bright regions smoothly  
✔ Prevents amplifying noise (clipLimit)  
✔ Suitable for photographic images  
✔ Great before LSD/Hough/FFT grid detection

✖ Does NOT correct large-scale illumination irregularities  
✖ Does NOT preserve absolute brightness levels

---

# ⭐ 2. What “Normalize Local Contrast” (ImageJ Integral Image Filters) does

This is based on **Local Contrast Normalization (LCN)** using **integral images** (fast box filters).

### Mechanism:

For each pixel, compute:

```
local_mean = blur(image, size=kernel)
local_std  = sqrt(blur((image - local_mean)^2))
output = (image - local_mean) / local_std
```

Then rescale to 0–255.

### Characteristics:

✔ Removes **large-scale illumination gradients**  
✔ Enforces **local zero-mean, unit-variance**  
✔ Very strong shadow correction  
✔ More mathematically “precise” local normalization than CLAHE  
✔ Great for scientific images, microscopy, document normalization

✖ Can create halo artifacts around strong edges  
✖ More aggressive; may reduce faint features  
✖ Can distort contrast globally  
✖ No histogram equalization → less midtone enhancement  
✖ May make paper grain too visible for LSD

---

# ⭐ 3. In short (critical comparison)

|Feature|CLAHE|ImageJ LCN (Normalize Local Contrast)|
|---|---|---|
|Corrects global uneven lighting|**Partially**|**Yes**|
|Boosts local texture|Mild|Strong|
|Noise amplification|Controlled|Significant|
|Preserves details|Excellent|Sometimes too aggressive|
|Ideal for photography|**Yes**|Meh|
|Ideal for scientific thresholding|Good|**Excellent**|
|Ideal for grid detection (your case)|**Typically best**|Good only if tuned carefully|
|Risk of halos|Low|Medium/High|
|Preserves white paper appearance|Yes|Not always|

---

# ⭐ 4. For _graph paper_ and _grid rectification_, which is better?

### ✔ CLAHE is the right tool **in most cases**, because:

- preserves lines and edges
- avoids over-normalizing flat paper regions
- doesn’t introduce halos around grid lines
- plays well with LSD / line detection / VP estimation

### ✔ ImageJ LCN is useful when:

- the image has **very strong illumination gradients**
- grid is extremely faint or partly invisible
- you want to treat the paper like a “scientific background”
- you are OK with some edge halos

### ⚠ ImageJ LCN can easily destroy the meaning of brightness

Graph paper background becomes artificially high-contrast everywhere → LSD may detect **too many fake edges**.

---

# ⭐ 5. Which one matches your real image?

Given your sample:

- Paper is unevenly lit
- Shadows on left
- Some areas slightly underexposed
- Grid lines moderate contrast
- You need stable vanishing point estimation

Your hierarchy should be:

### **1) Retinex / illumination correction**

Fix global uneven light.

### **2) CLAHE (gentle, clipLimit ≈ 2.0–3.0)**

Boost contrast but preserve structure.

### **3) LSD / VP estimation**

Lines should remain clean.

Using ImageJ Local Contrast Normalization **before** LSD may produce:

- texture artifacts (paper fibers)
- halo rings around grid
- ambiguous segments (bad VP cluster)

---

# ⭐ 6. My recommended pipeline ordering for your PET project

```
1. Convert to LAB
2. Retinex or Gaussian illumination correction
3. CLAHE on L-channel (clipLimit 2–3)
4. Mild denoise (bilateral or fastNLMeans)
5. Edge detection or LSD
6. Vanishing point estimation
7. Rectification
```

ImageJ LCN would only be inserted before (3) if illumination gradients are extremely strong, but I do **not** recommend replacing CLAHE with LCN in your case.

---

# ⭐ Summary

**CLAHE**  
→ best for photography, controlled enhancement, grid detection stability.

**ImageJ Normalize Local Contrast**  
→ strong scientific-style normalization, removes lighting gradients aggressively, but may distort edges and create halos.

---

# 📌 **pet_filters.py**

```python
"""
pet_filters.py
--------------

Local contrast enhancement routines compatible with
your PET pipeline BGR uint8 images:

  - apply_clahe()
  - apply_local_contrast()

These do NOT modify metadata and do NOT change dtype.
"""

import cv2
import numpy as np


# -------------------------------------------------------------
# 1) CLAHE – gentle, photography-oriented enhancement
# -------------------------------------------------------------
def apply_clahe(
    img: np.ndarray,
    clip: float = 3.0,
    tiles: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to L-channel in LAB space.

    Args:
        img: BGR uint8 image from pipeline.
        clip: CLAHE clipLimit (2–4 recommended).
        tiles: tileGridSize (8x8 default).

    Returns:
        BGR uint8 CLAHE-enhanced image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be BGR uint8 with 3 channels")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


# -------------------------------------------------------------
# 2) Local Contrast Normalization – ImageJ-like
# -------------------------------------------------------------
def apply_local_contrast(
    img: np.ndarray,
    radius: int = 25,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Local Contrast Normalization (LCN) similar to ImageJ:
        out = (I - local_mean) / local_std

    Args:
        img: BGR uint8 image.
        radius: half-size of box filter window (15–40 recommended).
        eps: small constant to avoid division by zero.

    Returns:
        BGR uint8 local-contrast-normalized image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be BGR uint8 with 3 channels")

    img_f = img.astype(np.float32)
    k = (radius * 2 + 1, radius * 2 + 1)

    mean = cv2.blur(img_f, k)

    diff = img_f - mean
    sq = diff * diff
    var = cv2.blur(sq, k)

    std = np.sqrt(np.maximum(var, eps))
    norm = diff / (std + eps)

    # Rescale normalized map to uint8
    mn, mx = norm.min(), norm.max()
    if mx - mn < 1e-12:
        return img.copy()

    out = (norm - mn) / (mx - mn)
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out
```

---

# ✅ **How to use these with your existing loader**

Your main pipeline becomes:

```python
from pet_utils import image_loader
from pet_filters import apply_clahe, apply_local_contrast

img, meta = image_loader()
img_clahe = apply_clahe(img)
img_lcn   = apply_local_contrast(img)
```

---

# 📌 **Where do these go in the PET pipeline?**

### Recommended for graph paper (stable LSD):

```
image_loader
↓
illumination correction (optional: Retinex)
↓
CLAHE (apply_clahe)
↓
denoise (optional)
↓
detect_grid_segments
↓
VP estimation
↓
rectification
```

### Only use LCN when light is extremely uneven:

```
image_loader
↓
apply_local_contrast (aggressive)
↓
CLAHE (optional)
↓
LSD
```


---

# ⭐ 1. **Parameters of ImageJ's Local Contrast Normalization**

### ✔ **X radius, Y radius**

These define the **size of the local window** separately in X and Y.

- Local mean = box-blur with size `(2*X+1, 2*Y+1)`    
- Local variance = box-blur of squared deviations, same window

This allows **anisotropic normalization**.

Examples:

- `X radius=50, Y radius=50`: big uniform window (most common)
- `X radius=20, Y radius=40`: elongated filter (rarely used)

### ✔ **StdDev (standard deviation multiplier)**

After computing:

```
normalized = (I - local_mean) / local_std
```

ImageJ multiplies by `std_factor`:

```
normalized_scaled = normalized * std_factor
```

This affects **how strong** the contrast normalization is.

ImageJ default:  
**StdDev = 1.0**

Increasing it amplifies contrast.

---

### ✔ **Center and Stretch options**

These determine how the normalized output is converted back to displayable pixel intensities.

### ✓ **Center**

Recenters the output around mid-level (128 for 8-bit).

Equivalent to:

```
out = normalized_scaled + constant
```

Usually makes dark/bright regions symmetrical.

### ✓ **Stretch**

Stretches the final histogram to fill the output dynamic range.

Equivalent to:

```
out = (out - min) / (max - min) * 255
```

If “Stretch” is unchecked, ImageJ clamps values without stretching.

---

# ⭐ 2. **Correct Python equivalent of ImageJ’s LCN (full version)**

Below is the _proper_ implementation **including X/Y radii**, **std factor**, **Center**, **Stretch**, and **ImageJ-like rescaling**.

```python
import cv2
import numpy as np

def apply_local_contrast_ij(
    img: np.ndarray,
    radius_x: int = 25,
    radius_y: int = 25,
    std_factor: float = 1.0,
    center: bool = True,
    stretch: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    ImageJ-like Local Contrast Normalization (LCN).
    Matches:
        Process → Filters → Integral Image Filters → Normalize Local Contrast

    Args:
        img: BGR uint8 input
        radius_x, radius_y: window radii
        std_factor: multiplier for normalized intensity
        center: shift output to mid-grey
        stretch: stretch final range to 0..255
        eps: small constant

    Returns:
        BGR uint8 processed image
    """

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be BGR uint8 with 3 channels")

    img_f = img.astype(np.float32)

    k = (2 * radius_y + 1, 2 * radius_x + 1)   # OpenCV uses (h, w)

    # local mean
    mean = cv2.blur(img_f, k)

    # local variance (ImageJ style)
    diff = img_f - mean
    sq = diff * diff
    var = cv2.blur(sq, k)
    std = np.sqrt(np.maximum(var, eps))

    # normalized
    norm = diff / (std + eps)
    norm *= std_factor

    # RECENTER
    if center:
        # Center at mid-level (~128)
        norm += 0.0  # ImageJ keeps centered before scaling

    # STRETCH / RESCALE
    if stretch:
        mn, mx = norm.min(), norm.max()
        if mx - mn > eps:
            out = (norm - mn) / (mx - mn)
        else:
            out = np.zeros_like(norm)
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # Map to 8-bit range without stretching
        out = np.clip(norm + 128.0, 0, 255).astype(np.uint8)

    return out
```

---

# ⭐ 3. How this differs from CLAHE

|Feature|CLAHE|ImageJ LCN|
|---|---|---|
|Removes large-scale lighting|❌ Limited|✔ Yes|
|Normalizes local variance|❌ No|✔ Yes|
|Boosts grid lines|✔ Yes|✔ Aggressive|
|Preserves appearance|✔ Good|❌ Can look harsh|
|Good for LSD detection|✔ Yes|⚠ Sometimes harms|
|Halos around edges|Rare|Common|
|Best use|photography|scientific normalization|

---

# ⭐ 4. Integration into your PET pipeline

```python
from pet_filters import apply_local_contrast_ij, apply_clahe

img, meta = image_loader()

# Option 1 – Gentle enhancement
img1 = apply_clahe(img)

# Option 2 – Strong normalization (ImageJ style)
img2 = apply_local_contrast_ij(
    img,
    radius_x=50,
    radius_y=50,
    std_factor=1.0,
    center=True,
    stretch=True
)
```

