# **Parameter Ranges**

Below is a **complete, calibrated table of sensible ranges** for _every_ control in your camera-correction engine — designed specifically so you can **sample uniformly/randomly** from:

- **0 → subtle → realistic → pronounced-but-not-crazy maximum**    
- while **never producing pathological/obviously-fake results**
- and preserving **forensic plausibility** across all artifacts.

This table is based on:  
✔ physical camera behavior  
✔ smartphone ISP curves  
✔ compact camera characteristics  
✔ realistic noise budgets  
✔ typical lens distortion magnitudes  
✔ real-world JPEG quantization ranges  
✔ actual CFA artifacts seen in cheap sensors  
✔ your forensic goals (avoid “tells”)

You can plug these directly into your random generator.

---

## ✅ MASTER TABLE — Sensible Ranges for Randomized Sampling

Each entry is:

**[min, typical, strong, max-safe]**

Plus a short explanation.

---

## 1. **Optics / Lens Geometry**

### 🔸 Radial Distortion: `k1`

| Level                | Value   |
| -------------------- | ------- |
| Disabled             | `0.0`   |
| Subtle               | `±0.05` |
| Realistic smartphone | `±0.12` |
| Max safe             | `±0.20` |

**Range for sampling:**  
➡ `k1 ∈ U(-0.20, 0.20)` with optional bias toward near-zero.

---

### 🔸 Radial Distortion: `k2`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`±0.005`|
|Realistic|`±0.01`|
|Max safe|`±0.02`|

**Range for sampling:**  
➡ `k2 ∈ U(-0.02, 0.02)`  
IMPORTANT: `k2` must be ~10× smaller than `k1`.

---

### 🔸 Rolling Shutter Skew: `rolling_strength`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.01`|
|Realistic phone video|`0.03`|
|Max safe|`0.05`|

**Range:**  
➡ `rolling_strength ∈ U(0.0, 0.05)`

---

## 2. **CFA & Demosaicing**

CFA artifacts depend on resolution, but strength is not explicitly exposed.  
You may add a switch:

- `"cfa": True/False`
    
- Probability: 0.7 True (recommended)
    

CFA artifacts should be ON most of the time because all non-RAW cameras have them.

---

## 3. **Sensor Noise: PRNU + FPN + Shot + Read**

These scale with ISO.

We express ranges **at ISO scale = 1.0**.  
Multiply by ISO_SF = {0.6, 1.0, 1.6}.

---

### 🔸 PRNU: `base_prnu_strength`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.001`|
|Realistic|`0.003–0.006`|
|Max safe|`0.010`|

**Range:**  
➡ `base_prnu_strength ∈ U(0.0, 0.010)`

Phone sensors can hit 0.007; above 0.01 looks broken.

---

### 🔸 Row FPN: `base_fpn_row`

### 🔸 Column FPN: `base_fpn_col`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.001`|
|Realistic|`0.002–0.004`|
|Max safe|`0.010`|

**Range:**  
➡ `base_fpn_row ∈ U(0.0, 0.010)`  
➡ `base_fpn_col ∈ U(0.0, 0.010)`

Notes:

- Row-band is more common than col-band.
    
- You may bias row-band ▷ col-band.
    

---

### 🔸 Shot Noise: `base_shot_noise`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.005`|
|Realistic|`0.012–0.020`|
|Max safe|`0.030`|

**Range:**  
➡ `base_shot_noise ∈ U(0.0, 0.030)`

Shot noise >0.04 on float-linear signals begins to resemble RAW ISO 6400–12800, which is not typical for lab photos.

---

### 🔸 Read Noise: `base_read_noise`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.001`|
|Realistic|`0.002–0.004`|
|Max safe|`0.008`|

**Range:**  
➡ `base_read_noise ∈ U(0.0, 0.008)`

Phones rarely exceed 0.005 unless the exposure is extremely low.

---

## 4. **ISP Denoising & Sharpening**

### 🔸 `denoise_strength`

Controls smoothing + bilateral kernel scale.

| Level                | Value       |
| -------------------- | ----------- |
| Disabled             | `0.0`       |
| Subtle               | `0.10`      |
| Realistic smartphone | `0.20–0.50` |
| Strong but plausible | `0.60–0.80` |
| Max safe             | `1.0`       |

**Sampling Range:**  
➡ `denoise_strength ∈ U(0.0, 1.0)`

Smartphones typically run between 0.20–0.60 depending on ISO.

---

### 🔸 `blur_sigma`

Radius of Gaussian blur used for unsharp mask.

| Level               | Value     |
| ------------------- | --------- |
| Disabled (not used) | n/a       |
| Subtle              | `0.5`     |
| Realistic phone     | `0.8–1.5` |
| Strong halos        | `2.0–2.5` |
| Max safe            | `3.0`     |

**Range:**  
➡ `blur_sigma ∈ U(0.5, 3.0)`

Notes:

- Smaller sigma = tighter micro-contrast
- Larger sigma = more aggressive halo shaping (Pixel-like clarity)

---

### 🔸 `sharpening_amount`

| Level                | Value     |
| -------------------- | --------- |
| Disabled             | `0.0`     |
| Low                  | `0.2`     |
| Realistic smartphone | `0.5–0.8` |
| Max safe             | `1.0`     |

**Range:**  
➡ `sharpening_amount ∈ U(0.0, 1.0)`

Beyond 1.0 halos look extremely artificial.

---

## 5. **Tone Mapping**

Two parameters:

- `tone_strength` — compression of highlights & shadows
    
- `scurve_strength` — midtone contrast pop
    

### 🔸 Tone Strength: `tone_strength`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.20`|
|Realistic phone|`0.40–0.65`|
|Max safe|`0.80`|

**Range:**  
➡ `tone_strength ∈ U(0.0, 0.80)`

> > 0.85 starts to “HDR-burn” the image.

---

### 🔸 S-curve Contrast: `scurve_strength`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.10`|
|Realistic phone|`0.20–0.30`|
|Max safe|`0.40`|

**Range:**  
➡ `scurve_strength ∈ U(0.0, 0.40)`

> > 0.5 looks like Instagram filters.

---

## 6. **Vignetting & Color Warmth**

### 🔸 Vignette: `vignette_strength`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.10`|
|Realistic compact|`0.20–0.35`|
|Max safe|`0.50`|

**Range:**  
➡ `vignette_strength ∈ U(0.0, 0.50)`

---

### 🔸 Color Warmth: `color_warmth`

|Level|Value|
|---|---|
|Disabled|`0.0`|
|Subtle|`0.02`|
|Realistic|`0.05–0.12`|
|Max safe|`0.20`|

**Range:**  
➡ `color_warmth ∈ U(0.0, 0.20)`

Warmth beyond 0.2 starts to look like filters rather than sensors.

---

## 7. **JPEG Compression**

### 🔸 JPEG Quality: `jpeg_quality`

|Level|Compression|Quality|
|---|---|---|
|Disabled|none|N/A|
|Subtle artifacts|mild|92–98|
|Realistic phone|medium|85–92|
|Strong|visible but plausible|70–85|
|Max safe|borderline|60|

**Range for sampling (safe):**  
➡ `jpeg_quality ∈ U(70, 98)`  
(Skip 60 unless intentionally producing harsh artifacts.)

---

## 8. **ISO Scaling**

You already have:

```
ISO_SF = {"low": 0.6, "mid": 1.0, "high": 1.6}
```

These are excellent.

If you want continuous ISO:

➡ `iso_level ∈ LogUniform(0.5, 2.0)`

---

## ✔ COMPLETE SUMMARY TABLE (COPY/PASTE)

| Parameter            | Range for random sampling          | Notes                                                          |
| -------------------- | ---------------------------------- | -------------------------------------------------------------- |
| `k1`                 | `[-0.20, 0.20]`                    | main lens distortion                                           |
| `k2`                 | `[-0.02, 0.02]`                    | weaker quartic term                                            |
| `rolling_strength`   | `[0.0, 0.05]`                      | rolling shutter skew                                           |
| `base_prnu_strength` | `[0.0, 0.010]`                     | pixel-level variations                                         |
| `base_fpn_row`       | `[0.0, 0.010]`                     | row banding                                                    |
| `base_fpn_col`       | `[0.0, 0.010]`                     | column banding                                                 |
| `base_shot_noise`    | `[0.0, 0.030]`                     | brightness-dependent                                           |
| `base_read_noise`    | `[0.0, 0.008]`                     | additive Gaussian                                              |
| `denoise_strength`   | `0.0 → 1.0`                        | Controls bilateral filter strength (0 = off, 1 = very strong). |
| `blur_sigma`         | `0.5 → 3.0`                        | Radius of Gaussian blur for unsharp mask.                      |
| `sharpening_amount`  | `[0.0, 1.0]`                       | halo strength                                                  |
| `tone_strength`      | `[0.0, 0.80]`                      | highlight compression                                          |
| `scurve_strength`    | `[0.0, 0.40]`                      | midtone contrast                                               |
| `vignette_strength`  | `[0.0, 0.50]`                      | optical falloff                                                |
| `color_warmth`       | `[0.0, 0.20]`                      | slight warm tint                                               |
| `jpeg_quality`       | `[70, 98]`                         | camera-compression artifacts                                   |
| `iso_level`          | `{low, mid, high}` or `[0.5, 2.0]` | noise multiplier                                               |

---
