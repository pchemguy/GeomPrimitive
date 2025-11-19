# **Fiji ImageJ Setup**

- [ImageJ 1.54p](https://downloads.imagej.net/fiji/stable/fiji-stable-win64-jdk.zip) from https://fiji.sc
- [IJ-Retinex](https://github.com/fiji/Fiji_Plugins/blob/main/src/main/java/Retinex_.java). Note, the source then compiled with [script](./Compile/build_retinex.bat)

## **Processing File**

photo_2025-11-17_23-50-02.jpg (see WhatsApp)

## **Retinex Settings**

| Level   | Scale | Scale division | Dynamic | Comment                                                                                                                  |
| ------- | :---: | :------------: | :-----: | ------------------------------------------------------------------------------------------------------------------------ |
| Uniform |  240  |       3        |  1.20   | Defaults. More even lighting by slight darkening of the top brighter region, better grid contrast                        |
| Low     |  240  |       3        |  1.20   | Blueish glow around the bio sample (dark); glow around dark marker cap as well, improved contrast, higher noise as well. |
| Uniform |  16   |       3        |  1.20   | Cartoonish, high contrast, increased noise, glow around the dark bio sample and dark marker cap.                         |
| Low     |  250  |       3        |  1.00   | Aggressively evened, cartoonish, high contrast, noise, glow around dark parts (sample and marker)                        |

## **Interesting Features**

### Plugins -> Integral Image Filters -> Normalize Local Contrast


# 🚀 **Retinex Parameter Guide (Practical & Correct)**


https://chatgpt.com/c/691c227e-1018-832c-9eb6-54ddc6686dc3

Below is a **practical explanation** of the 4 parameters used in the classic Retinex plugin (`Retinex_`), **specifically in the context of correcting uneven lighting / illumination falloff / bright spots**.

These parameters come straight from the **original Gagnon & Faber Retinex implementation** used in older ImageJ builds.


The plugin exposes:

|Parameter|Meaning|Typical Range|Default|
|---|---|---|---|
|**Level**|Low-frequency emphasis (dark region boost)|Uniform / Low / High|Uniform|
|**Scale**|Gaussian surround radius (dynamic range equalization strength)|16–250|240|
|**Scale division**|Number of multi-scale layers|1–8|3|
|**Dynamic**|Contrast gain / compression factor|0.04–4|1.2|

Let’s interpret each parameter exactly as the algorithm uses them.

---

# 1. **LEVEL**

_(distribution of “gain” across intensities)_

This adjusts how much the algorithm enhances **dark areas** relative to bright ones.

### ✔ **Uniform**

- Applies the same gain across all intensities
- Natural and stable
- Good general-purpose correction
- **Best starting point for uneven lighting**

### ✔ **Low**

- More aggressive boost to **shadows**
- Great for images where corners or edges are very dark
- Helps “flatten” illumination differences
- Can introduce noise

### ✔ **High**

- Boosts **bright regions** more than dark
- Rarely useful for uneven lighting correction
- Use only if highlights need more detail

▶ **Recommendation:**  
Start with **Uniform**, try **Low** if edges remain dark.

---

# 2. **SCALE**

_(Gaussian surround radius — determines how much illumination is smoothed away)_

This is the **most important parameter** for uneven lighting.

It controls the size of the “ambient illumination” Retinex tries to subtract.

### ✔ Small scale (16–60)

- Removes very local brightness variations
- Sensitive to texture
- Can cause halos and artifacts
- **Not recommended** for global uneven lighting

### ✔ Medium scale (80–160)

- Removes moderate illumination gradients
- Best for typical images with light falloff
- Balanced and stable results

### ✔ Large scale (180–250)

- Treats illumination as a slowly varying field
- Excellent for:
    - vignetting correction
    - images with bright spot in center
    - uneven illumination from a desk lamp
- Very smooth and natural

▶ **Recommendation:**  
For uneven lighting correction start at **200–240**  
This captures broad illumination gradients.

---

# 3. **SCALE DIVISION**

_(Number of multi-scale Retinex layers)_

This runs Retinex at multiple scales:

- 1 layer = single-scale Retinex
- 3 layers = multi-scale retinex (classic MSR)
- More layers = slightly more detail equalization

### ✔ 1

- Fast, but harsh
- Looks artificial
- Only local normalization

### ✔ 3

- Balanced
- Smooth gradients
- **Recommended default**

### ✔ 5–8

- Very fine detail enhancement
- Can lift noise
- Only helpful in photos, rarely in scientific images

▶ **Recommendation:**  
Use **3** for almost all uneven-lighting problems.

---

# 4. **DYNAMIC**

_(contrast compression factor)_

Controls how much the output range is stretched or compressed **after** Retinex normalization.

### ✔ Lower values (0.04–1.0)

- Softer contrast
- Good if uneven illumination creates harsh transitions
- Helps prevent halos

### ✔ Default (1.2)

- Balanced
- Corrects contrast without blowing out highlights

### ✔ Higher (2–4)

- Very aggressive
- Strong local contrast
- Rarely helpful for uneven lighting
- Mainly useful for enhancing faint detail

▶ **Recommendation:**  
Use **1.0–1.4** for illumination correction.

---

# 🎯 **How to Choose Parameters for Uneven Lighting Compensation**

Here is the best practice workflow:

### **Step 1 — Use Uniform level**

```
Level = Uniform
```

- Avoids unnatural local contrast distortions

### **Step 2 — Choose a LARGE scale**

```
Scale = 200–240
```

This removes broad low-frequency illumination patterns.

### **Step 3 — Use 3 multi-scale layers**

```
Scale division = 3
```

Balanced; consistent with multi-scale Retinex theory.

### **Step 4 — Keep Dynamic moderate**

```
Dynamic = 1.0–1.4
```

---

# 📌 Recommended Starting Preset for Uneven Lighting

```
Level = Uniform
Scale = 240
Scale division = 3
Dynamic = 1.2
```

This is exactly what you want for correcting:

- vignetting
- central bright spots
- uneven desk lamp illumination
- top/bottom edge darkening
- light gradients across documents
- global illumination non-uniformity

---

# 💡 If you want the strongest illumination flattening

Try:

```
Level = Low
Scale = 250
Scale division = 3
Dynamic = 1.0
```

This heavily boosts dark corners but still looks natural.
