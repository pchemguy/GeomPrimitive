https://chatgpt.com/c/69245626-cc60-8328-9f27-a1445ec85386

# **How to choose `k_len`?**

---

## **1. Empirical heuristic from grid frequency (simplest practical)**

Compute a rough grid pitch using **FFT of the gradient magnitude**:

- Compute `mag = sqrt(gx² + gy²)`    
- FFT → radial average
- Find frequency peaks corresponding to horizontal / vertical line spacing
- Convert frequency → pixel period

Then set:

```
k_len ≈ 0.4–0.7 × estimated_pitch
```

This works extremely well for printed grids, even under mild perspective.

---

## **2. Distance transform + local maxima**

1. After binarization, compute the **distance transform**.    
2. Analyse the histogram of distances along ridges.
3. Extract dominant run lengths of straight-line segments.
4. Use the mode or the 90th percentile as an estimate of line length.

---

# **3. Autotune by scanning `k_len` and scoring response**

Try `k_len` values in range e.g. `[10, 15, 20, 25, 30, 35, 40]`.

For each:

1. Perform horizontal opening.    
2. Measure:
    - total surviving white pixels
    - number of connected components
    - component elongation (aspect ratio)
3. Choose the `k_len` where:
    - number of horizontal line pixels stabilizes (plateau),
    - noise collapses but grid still survives.

This is reliable and fast.

---

# **4. Use Hough or LSD just to estimate spacing (hybrid method)**

You can use a **minimal run of LSD** or a **constrained Hough**:

- Detect ~20–40 “stronger” long segments (up to perspective).    
- Cluster slopes to separate horizontal vs vertical.
- Compute spacing from median nearest-neighbor distances.

Then again set:

```
k_len = 0.5 × pitch
```

This gives excellent results and is resilient even when binarization is noisy.

---

# **5. Learn `k_len` from grid density (robust image-statistics method)**

Define:

```
R(k_len) = (line_pixels_after_opening / total_binary_pixels)
```

As k_len increases:

- noise removes first → R goes up,    
- grid lines eventually break → R drops sharply.

The best k_len is in the **maximal plateau before the drop**.

This is similar to choosing a threshold using the elbow method.

---

# **6. Use scale-space morphology (advanced)**

Perform opening at several scales simultaneously (e.g., 10, 20, 30, 40).

Compute multiscale persistence of horizontal strokes.

Select the **scale at which the lines have maximal morphological stability**.

This is analogous to SIFT scale-space interest point detection, but for line structures.

---

# **7. Full probabilistic model (overkill, but ideal)**

Model grid likelihood as:

```
P(line survives | k_len)
```

with priors on grid pitch distribution.

Choose k_len that maximizes posterior probability of observed erosion patterns.

This requires calibration but is extremely accurate.

---

# 📌 **Recommendation**

The best tradeoff between speed, robustness, and implementation complexity:

### **Use FFT-based pitch estimation (Method 1)**

→ then choose

### **k_len = round(0.5 × estimated_pitch)**

This avoids scanning, avoids Hough, works on blurred grids, and is fully automatic.
