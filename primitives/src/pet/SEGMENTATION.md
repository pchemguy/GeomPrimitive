https://gemini.google.com/app/1005d073e1f7b62d

> [!NOTE]
> 
> **For best results, consider combining Chroma + Saturation (possibly with Hue) + a masks, following by morphological modifications.**

The technical term for "distance from gray" is **Saturation** (or sometimes **Chroma**).

In color theory, "Gray" is defined as the state where $Red = Green = Blue$. As these values diverge from one another, the pixel becomes more "colorful."

Here are the three best ways to calculate this on a 0 to 1 scale, depending on whether you care about the brightness of the pixel or just the raw color purity.

---

### Method 1: HSV Saturation (The Standard)

This is the most common method used in computer vision (like OpenCV) and image editing software. It calculates "grayness" relative to the brightness of the pixel.

- **Logic:** It looks at the spread between the strongest and weakest color channels, scaled by the strongest channel.    
- **The Scale:**
    - **0:** Pure Gray (e.g., White, Black, or Gray).
    - **1:** Pure Color (e.g., Bright Red).

The Formula:

First, normalize your RGB values to the range $[0, 1]$.

Let 1$C_{max} = \max(R, G, B)$ and 2$C_{min} = \min(R, G, B)$.3

$$Saturation = \begin{cases} 0, & \text{if } C_{max} = 0 \\ \frac{C_{max} - C_{min}}{C_{max}}, & \text{otherwise} \end{cases}$$

> **Note:** This method considers Dark Red $(0.1, 0, 0)$ to be just as "saturated" (1.0) as Bright Red $(1, 0, 0)$. If you want dark colors to score lower, use Method 2.

### Method 2: Chroma (Absolute Distance)

If you want to know how much color exists **ignoring** scaling, use Chroma. This is often better if you consider "dark" colors to be closer to gray (black) than bright neon colors.

- **Logic:** Simply the range of the color channels.
- **The Scale:**
    - **0:** Perfect Gray.
    - **1:** Only achievable by pure, bright primary/secondary colors (e.g., Yellow $(1, 1, 0)$ or Blue $(0, 0, 1)$).

The Formula:

$$Chroma = C_{max} - C_{min}$$

### Method 3: Euclidean Distance from the Gray Line

This is the most mathematically robust "geometric" approach. In a 3D RGB cube, perfect grays lie on the diagonal line where 4$x=y=z$.5 This calculates the shortest distance from the pixel to that line.

- **Logic:** Strictly geometric distance in 3D space.
- **The Scale:** Requires normalization to fit 0-1.

The Formula:

$$Distance = \frac{\sqrt{(R-G)^2 + (R-B)^2 + (G-B)^2}}{\sqrt{2}}$$

_(Note: The denominator $\sqrt{2}$ normalizes the result so that pure Red $(1,0,0)$ results in 1.0)_.

---

### Python Implementation

Here is how you would implement the **HSV Saturation** (Method 1) and **Chroma** (Method 2) in Python.

Python

```
def calculate_gray_distance(r, g, b, method='hsv'):
    """
    Inputs: r, g, b are integers 0-255 or floats 0-1.
    Output: Float between 0.0 (gray) and 1.0 (colorful).
    """
    # Normalize to 0-1 if inputs are 0-255
    if r > 1 or g > 1 or b > 1:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    if method == 'chroma':
        # Method 2: Pure difference
        return delta
    
    elif method == 'hsv':
        # Method 1: Standard Saturation
        if c_max == 0:
            return 0.0
        return delta / c_max

# Example: A semi-grayish red pixel (150, 100, 100)
# Scaled: (0.58, 0.39, 0.39)
# Result HSV: 0.33 (33% saturated)
```

### Recommendation

- Use **Method 1 (HSV)** if you are filtering images and want to detect color regardless of lighting conditions (e.g., detecting a red ball in a shadow).
- Use **Method 2 (Chroma)** if you are doing quality control or noise reduction and want to treat dark pixels as "less colorful" than bright pixels.

---

**S** (Saturation) from the **HSV** model is identical to **Method 1** in my previous response.

Here is the breakdown of why they are the same, and how they differ from other "S" values you might see (like in HSL).

### 1. The Match (HSV Saturation)

In the standard HSV (Hue, Saturation, Value) model, Saturation is defined as the "ratio of the difference to the maximum value."1

- **HSV Formula:** 2$S_{HSV} = \frac{Max(RGB) - Min(RGB)}{Max(RGB)}$
- **My Method 1 Formula:** $\frac{C_{max} - C_{min}}{C_{max}}$

These are mathematically identical. This is the definition used by **OpenCV**, **Python's `colorsys`**, and standard image processing libraries.

### 2. The Confusion (HSL Saturation)

Be careful not to confuse this with **S from HSL** (Hue, Saturation, Lightness), which is often used in CSS and web design. They share the same name ("Saturation") but calculate the value differently.

- **HSL Saturation** scales based on how close the color is to 50% lightness.3
- **HSV Saturation** scales based on the brightness of the strongest color channel.

Example Difference:

Imagine a very dark red pixel (50, 0, 0).
- **HSV Saturation:** **1.0 (100%)**.4 It considers this "pure red," just under low light.
- **HSL Saturation:** **1.0 (100%)**. It also sees this as pure.
- **My Method 2 (Chroma):** **~0.2**. It sees this as "close to black," so it gives it a low score.

### Summary Table

|**Term**|**Formula / Logic**|**Matches Previous Answer?**|
|---|---|---|
|**HSV Saturation**|$\frac{Max - Min}{Max}$|**Yes (Method 1)**|
|**HSL Saturation**|$\frac{Max - Min}{1 -|2L - 1|
|**Chroma**|$Max - Min$|**Yes (Method 2)**|

If you are using this for computer vision (like detecting objects), **S from HSV (Method 1)** is usually the correct choice because it reliably identifies a color (like a red stop sign) even if shadows make the pixels dark.

---

In Adobe Photoshop, HSV is labeled as **HSB** (Hue, Saturation, **Brightness**).

It is the exact same model (Brightness = Value), just a different name.

Here are the three best ways to view these values, depending on your workflow:

### 1. The Info Panel (Best for Hovering)

This is the fastest way to inspect pixels without clicking on them.

1. Go to `Window > Info` to open the panel.
2. Click the small **eyedropper icon** inside the Info panel (next to the RGB/CMYK values).
3. Select **HSB Color** from the dropdown list.
4. **Result:** Now, whenever you hover your mouse over any part of your image, the panel will show the live H, S, and B percentages for that specific pixel.

### 2. The Color Picker (Best for Analysis)

If you want to analyze a specific color in depth:
1. Click your **Foreground Color** swatch (bottom of the toolbar).
2. In the dialog that pops up, look at the buttons labeled **H**, **S**, and **B**.
3. **Result:** The numerical values are listed there.
    - **S:** Saturation (0–100%)
    - **B:** Brightness/Value (0–100%)

### 3. The Color Panel (Best for Sliders)

If you want to paint or design using this scale:
1. Go to `Window > Color`.
2. Click the **hamburger menu** (three lines) in the top-right corner of the panel.
3. Select **HSB Sliders**.
4. **Result:** Your color sliders will now operate on the HSV scale instead of RGB.

### ⚠️ Important Warning: HSL vs. HSB

Be careful if you use the **Hue/Saturation Adjustment Layer** (`Image > Adjustments > Hue/Saturation`).
- This tool uses **HSL** (Lightness), NOT **HSB** (Brightness).
- As we discussed, **Lightness** treats White as "0% Saturation," whereas **Brightness** treats White as "100% Value / 0% Saturation."
- **Rule of Thumb:** For pixel measurement, trust the **Info Panel (HSB)**. For color correction, you are stuck with the Hue/Saturation tool (HSL).

---

If you want to create a layer that **visually represents** this saturation math (turning your image into a "Heat map" where White = Pure Color and Black = Gray), you cannot do it with a simple Adjustment Layer.

You need to use a specific filter that decomposes the image data.

Here is the "Scientific" way to turn the **HSV Saturation** value into a visible Black & White layer in Photoshop.

### Method 1: The HSB/HSL Filter (Exact Math)

This method literally converts the math we discussed into pixels.
- **Black Pixel:** Saturation is 0 (Gray).    
- **White Pixel:** Saturation is 1 (Pure Color).

1. **Duplicate your image** into a new layer (`Ctrl+J` / `Cmd+J`).
2. Go to `Filter > Other > HSB/HSL`.
3. Set **Input** to "RGB" and **Output** to "HSB". Click OK.
    - _Your image will look like a broken psychedelic mess. This is normal._
4. Go to your **Channels Panel** (next to Layers).
5. Click on the **Green Channel**.
    - _In this specific filter, Photoshop stores the "Saturation" data in the Green channel._
6. Press `Ctrl+A` (Select All) then `Ctrl+C` (Copy).
7. Go back to your **Layers Panel**, create a new empty layer, and paste (`Ctrl+V`).    

**You now have a "Saturation Map" layer.** You can use this as a mask, or just use it to analyze which parts of your image are mathematically "pure" color.

---

### Method 2: The "Difference" Trick (Live/Non-Destructive)

If you want a layer that updates live as you paint/edit, you can use Blending Modes to approximate the "Chroma" (Distance from Gray) method.

1. Create a **Black & White Adjustment Layer** at the top of your stack.
2. Change the **Blending Mode** of this adjustment layer to **Difference**.
3. **Result:**
    - The image turns mostly black.
    - Areas that are **Pitch Black** are perfectly neutral gray.
    - Areas that light up (become brighter) are "distant" from gray.

_Note: This calculates the raw distance (Chroma), not the scaled Saturation ratio, but it is excellent for quickly spotting "color contamination" in gray areas._

### Method 3: The Standard Editing Way

If you just want to control the S value we discussed:
1. Add a **Hue/Saturation Adjustment Layer**.
2. Use the **Saturation Slider**.
    - $-100$ sets $S=0$ (Gray).        
    - $+100$ pushes $S$ toward 1.

**Which one are you trying to achieve?** (Visualizing the data vs. just editing the colors?)

> [!NOTE] 
> 
> I visually see that adjusting both hue and saturation on the adjustment layer visually improves contrast between the white-grey background paper and saturated red biosample compared to saturation alone. (Saturation works, but adjusting hue appears to be helping further.)

So, when I change hue for a high S target, color changes significantly. For center - not so much

Think of the color wheel (like the image above) as a **merry-go-round**:
1. **At the Edge (High S):** You are far from the center. If you move 90 degrees (change Hue), you travel a massive distance—from Red all the way to Green. The change is obvious.
2. **At the Center (Low S):** You are hugging the center pole. If you spin 90 degrees (change Hue), you barely move an inch. You are still just "standing at the pole."

### The "Multiplier" Effect

Mathematically, you can think of Saturation as a **volume knob** for Hue.
- **Hue** is the song playing.
- **Saturation** is the volume.

If the volume is at 0 (Gray center), it doesn't matter if you switch the song from "Jazz" (Blue) to "Metal" (Red). You can't hear it. The difference is mathematically there, but perceptually invisible.

### Why this confirms your "Gray Imbalance" approach

This observation is exactly why your "Gray Imbalance" metric (Saturation) is the correct one, and why Hue is dangerous to use near the center.

- **At the Center (Gray):** Hue is unstable and meaningless. A tiny shift in pixel values (noise) flips the Hue wildy, but the color looks the same.
- **At the Edge (Color):** Hue is stable and meaningful.

---

This is a great observation. If adjusting Hue is improving your contrast, it means you have discovered a hidden variable in your image: **Perceived Brightness (Luminance).**

You are right: Saturation separates "Color vs. Gray."

But Hue separates "Bright Colors vs. Dark Colors."

Here is why shifting the Hue of your Red Biosample makes it pop against the Gray background.

### 1. The "Luma" Effect (The Human Eye)

Not all colors are created equal in brightness. Our eyes are very sensitive to Green/Yellow and very insensitive to Blue/Red.

- **Yellow/Green:** Perceived as very bright (Light).
- **Blue/Red:** Perceived as very dark (Heavy).

What is happening in your image:

If your Red sample and your Gray background have a similar brightness level, they might blend together even if the saturation is different.

- By shifting the **Hue** of the sample from **Red to Blue**, you artificially force the sample to become **darker** (in grayscale terms).
- By shifting from **Red to Yellow**, you force it to become **lighter**.

You are essentially using Hue to "hack" the contrast.

### 2. The "Dirty Gray" Factor

In real-world photography (biosamples on paper), "Gray" paper is almost never mathematically neutral.

- **The Reality:** Your white paper likely has a slight **Blue tint** (from shadows) or **Yellow tint** (from room lights).
- **The Adjustment:** By shifting the Master Hue, you are rotating _both_ the sample and the background tint. You might be rotating the background tint into a color that the eye ignores (like a subtle blue) while rotating the sample into a "screaming" color (like bright neon green).    

---

### The "Pro" Solution: Lab Color Mode

Since you are dealing with **Red vs. Gray**, there is a much more powerful way to separate them than using Hue/Saturation. It is called **Lab Color**.

In RGB, Red/Green/Blue are mixed together.

In Lab, Lightness (1$L$) is separated from Color (2$a$ and 3$b$).

- **Channel $a$:** The "Green vs. Red" axis.
- **Channel $b$:** The "Blue vs. Yellow" axis.

Why this is perfect for you:

Since your sample is Red and your background is Neutral, the $a$ channel contains almost 100% of the contrast information you need, with zero lighting noise.

#### How to use it in Photoshop for Quantification:

1. **Duplicate Image:** `Image > Duplicate`.
2. **Convert Mode:** `Image > Mode > Lab Color`.
3. **Open Channels:** Go to the **Channels Panel**.
4. **Click the "a" Channel:**
    - **Gray Background:** Will appear perfectly **Middle Gray (50%)**.
    - **Red Sample:** Will appear **Bright White** (or Dark Black, depending on the red tone).
    - **Result:** The contrast will be massive. The lighting/shadows (stored in the "L" channel) will disappear completely.

---

This is the classic "Same Color, Different Intensity" problem. You have hit the limit of what a single metric can do.

Your intuition to **intersect the masks** is 100% correct. This is how professional segmentation pipelines work: you use one mask to define "Where the stuff is" and another to define "What the stuff is."

Here is why your "Blood vs. Organ" problem exists and the logic for the intersection you proposed.

### The "Anatomy" of your Data

1. **The Organ:** High Saturation (Solid), Specific Hue (Red), High Density.
2. **The Blood Trails:** Medium Saturation (Translucent/Thin), Same Hue (Red), Low Density.
3. **The Background (Paper):** Low Saturation (Noise), Neutral Hue.

### Why the single channels fail

- **Lab 'a' (Redness):** Fails because it asks "Is it Red?" Both the organ and the blood scream "YES." It cannot distinguish between the two siblings.
- **Saturation (Purity):** Fails because it asks "Is it Vivid?" The background noise (paper grain, lighting artifacts) sometimes whispers "Yes," creating speckles.    

### The Solution: The "Logical AND" Intersection

You need to combine the strengths of both to cancel out the weaknesses.

The Formula:

$$Mask_{Final} = Mask_{Saturation} \times Mask_{LabA}$$

_(In image processing terms, multiplying two masks is a Logical AND operation. A pixel stays bright only if it is bright in BOTH masks.)_

Here is how the intersection fixes your specific issues:

| **Area**             | **Mask 1: Saturation**      | **Mask 2: Lab 'a'**  | **Result (Intersection)** |
| -------------------- | --------------------------- | -------------------- | ------------------------- |
| **Organ**            | **High** (It's solid color) | **High** (It's red)  | **Kept (Target)**         |
| **Blood Trail**      | **Low/Med** (It's thin)     | **High** (It's red)  | **Reduced/Removed**       |
| **Background Noise** | **Med** (Noise spikes)      | **Zero** (It's gray) | **Cleaned (Black)**       |

### How to execute this (The "Double Gating" Workflow)

Since you mentioned `S+H` works well for the organ but has noise, use **Lab 'a'** purely as a "Cleaning Filter" (a gatekeeper).

**In Photoshop:**

1. **Base Layer:** Your high-contrast **Saturation** map (or your S+H tweak). Ideally, this makes the organ pop, even if the background looks messy.
2. **Mask Layer:** A copy of the **Lab 'a'** channel.
3. **The Move:** Apply the Lab 'a' channel as a **Layer Mask** onto your Saturation layer.
    - _Result:_ The Saturation layer will be forced to become transparent wherever the image isn't Red. The background noise (which is not red) will vanish instantly.        

```
# Create the two masks
mask_chromaticity = saturation_channel  # Good for Organ vs Blood structure
mask_redness = lab_image[:,:,1]         # Good for cleaning background

# Normalize them to 0-1
mask_chromaticity = normalize(mask_chromaticity)
mask_redness = normalize(mask_redness)

# INTERSECTION (Multiplication)
# This suppresses anything that isn't BOTH "Vivid" and "Red"
final_mask = mask_chromaticity * mask_redness

# Optional: Power law to increase separation (Contrast boost)
final_mask = final_mask ** 2
```

### One Final Trick: The "Morphological Opening"

If "Blood Trails" are still lingering because they are just too similar to the organ, you can use a shape-based filter.

Since blood trails are usually thin (lines) and the organ is a blob (circle/mass):

1. Create your mask.
2. Apply **Morphological Erosion** (shaves a few pixels off all edges).
    - _The thin blood trails will disappear completely._
    - _The large organ will just get slightly smaller._
3. Apply **Morphological Dilation** (grows the pixels back).
    - _The organ returns to its original size._        
    - _The blood trails are gone forever (they can't grow back from nothing)._

---

These three tools—**Erosion/Dilation**, **Fill Holes**, and **Watershed**—are known as "Morphological Operations." They stop looking at _color_ (pixel values) and start looking at _shape_ (pixel geography).

Here is how to use them to solve your "Blood Trail vs. Organ" problem.

### 1. Erosion & Dilation (The "Opening" Operation)

This is the magic bullet for your blood trails. In image processing, performing an **Erosion** followed immediately by a **Dilation** is called **"Opening."**

- **The Logic:** Imagine the computer traces the outline of your white shapes and "peels off" a layer of pixels (like peeling an onion).
- **Step A: Erosion (The Peel):**
    - The **Organ** is fat. If you peel off 2 layers of pixels, it gets slightly smaller, but it is still there.
    - The **Blood Trail** is thin. If you peel off 2 layers of pixels, **it vanishes completely.** It ceases to exist.
- **Step B: Dilation (The Regrowth):**
    - Now you add 2 layers of pixels back to everything that survived.
    - The **Organ** grows back to its original size.
    - The **Blood Trail** is already dead; it cannot grow back from nothing.

In Fiji: Process > Binary > Open.

(If the trails are thick, run Erode manually a few times until they vanish, then run Dilate the same number of times).

### 2. Fill Holes (The Interior Fixer)

You mentioned that Saturation creates noise inside the organ (maybe glare spots or dry patches that look gray). This creates "holes" in your mask.

- **The Logic:** The algorithm looks for any black pixels that are completely surrounded by white pixels (like a lake inside an island) and forces them to white.
- **When to use it:** Use this **AFTER** Erosion/Dilation. If you use it before, you might accidentally fill in the gap between a blood trail and the organ, merging them into one giant blob.

**In Fiji:** `Process > Binary > Fill Holes`.

### 3. Watershed (The Separator)

Use this if the blood trail is physically **touching** the organ, or if two organs are touching each other.

- **The Logic:** Imagine the white parts of your image are mountains and the black parts are the sea. It simulates rain falling on the mountains.
    - Where the water flows down into two different "basins," it builds a dam (a 1-pixel black line) to separate them.
- **The Result:** It cuts the "neck" connecting two objects. If a thick blood trail is fused to the organ, Watershed will slice it off at the thinnest point.

**In Fiji:** `Process > Binary > Watershed`.

---

### The "Golden Recipe" for your Bio-Sample

Since you have a **Solid Target (Organ)** and **Thin Noise (Trails)**, here is the exact order of operations you should run in Fiji/Python to get a perfect quantification:

1. **Threshold:** Create your mask using the **Saturation + Lab Intersection** we discussed.
2. **Fill Holes:** Quick pass to make sure the Organ doesn't have holes inside it (so the next step doesn't eat it from the inside out).
3. **Open (Erode + Dilate):**
    - _Action:_ Erode until the blood trails disappear.
    - _Action:_ Dilate to restore the organ's volume.
    - _Result:_ Clean, isolated organ.
4. **Watershed:** (Optional) Run this only if the organ is touching the edge of the paper or another sample.
5. **Analyze Particles:** Now you can ask Fiji to measure the area.

---

For your specific case—distinguishing a **solid organ** from **faint blood trails**—**Chroma is likely better than Saturation.**

It will definitely provides extra benefits because it removes the "Equalizer" effect of Saturation.

Here is the breakdown of why Chroma might be the missing link for your segmentation.

### 1. The "Signal Strength" Difference

This is the most critical distinction for biological samples.

- **Saturation (S)** is a **Ratio**.1 It asks: "Of the light coming from this pixel, what percentage is colored?"
    - _Thick Organ:_ 90% Red. **High S**.
    - _Faint Blood Trail:_ 20% Red / 80% White reflection. **Low S**.
    - _Dark Shadow:_ 10% Red / 90% Black. **High S** (Mathematically).
    - **The Trap:** Saturation artificially boosts dark, noisy pixels. It treats a "whisper" of red in a shadow the same as a "shout" of red in the light.
- **Chroma (C)** is an **Absolute Amount**. It asks: "What is the raw magnitude of the color?"
    - _Thick Organ:_ Strong signal. **High Chroma**.
    - _Faint Blood Trail:_ Weak signal. **Low Chroma**.
    - _Dark Shadow:_ Weak signal. **Low Chroma**.

**Benefit for you:** Chroma naturally suppresses the background noise and the faint blood trails because they lack "energy." Saturation tries to "normalize" them, making them look more significant than they are.

### 2. The Math Proof

Since $Saturation = Chroma / Brightness$ (roughly), we can flip the equation:

$$Chroma = Saturation \times Brightness$$

By using Chroma, you are effectively saying: _"I want pixels that are BOTH colorful AND bright enough to matter."_

If your blood trails are translucent (faint), they likely have a lower Chroma value than the dense, opaque organ tissue.

### 3. How to test "Chroma" right now (Without writing code)

You can simulate Chroma in Photoshop or Fiji/ImageJ to see if it improves separation.

In Photoshop (The "Difference" Method):

This approximates Chroma perfectly.

1. Make two duplicates of your image.
2. Set the top layer to **"Lighten"** mode (This essentially keeps the $Max(R,G,B)$).
3. Set the middle layer to **"Darken"** mode (This essentially keeps the $Min(R,G,B)$).
4. Set the top layer's blend mode to **"Difference"**.
5. **Result:** You are now looking at $(Max - Min)$, which is the exact formula for **Chroma**.

**Compare this visual to your Saturation layer.**

- You should see the **Organ** staying bright white.
- You should see the **Blood Trails** fading into dark gray (because their color difference is weaker).
- You should see **Dark Noise** turning pitch black.

### Summary

- **Use Saturation (S)** if you want to detect **ALL** red, regardless of how faint or dark it is (e.g., detecting microscopic trace elements).
- **Use Chroma (C)** if you want to detect **SIGNIFICANT** red (e.g., the main organ mass) and ignore "weak" red signals (trails and noise).

**My prediction:** For your "hottest area" separation, **Chroma** combined with the **Morphological Opening** (Erosion) will give you the cleanest mask.

---

There isn't a single button for "Chroma" in Photoshop, but since we know the formula is $Max(R,G,B) - Min(R,G,B)$, we can construct it manually using **Channels** and **Blending Modes**.

This will effectively create a "Chroma Map" where the brightness of the pixel equals its raw color intensity.

Here is the step-by-step recipe to build a Chroma visualization layer.

### The "Max minus Min" Technique

You need to create two "super-layers": one representing the highest value of every pixel ($Max$) and one representing the lowest ($Min$).

#### Phase 1: Create the "Max" Layer

1. Go to the **Channels Panel**.
2. Click the **Red** channel $\rightarrow$ `Ctrl+A` (Select All) $\rightarrow$ `Ctrl+C` (Copy).
3. Go back to **Layers Panel**, create a new layer, and `Ctrl+V` (Paste). Name it "Red".
4. Repeat this for **Green** and **Blue** (Paste them as new layers on top of "Red").
5. Select the top two layers (Blue and Green) and set their **Blending Mode** to **Lighten**.
    - _Logic:_ Photoshop compares the stacked pixels and keeps only the brightest one. You now have $Max(R,G,B)$.
6. Select all 3 layers (Red, Green, Blue), right-click, and choose **Merge Layers** (or `Ctrl+E`).
7. Rename this merged layer **"MAX Values"**.

#### Phase 2: Create the "Min" Layer

1. Repeat the exact same copy-paste process (Copy Red, Green, and Blue channels into 3 new layers).
2. Select the top two layers and set their **Blending Mode** to **Darken**.
    - _Logic:_ Photoshop compares them and keeps the darkest pixel. You now have $Min(R,G,B)$.
3. Select these 3 layers and **Merge Layers**.
4. Rename this merged layer **"MIN Values"**.

#### Phase 3: The Subtraction (Chroma)

1. Move the **"MAX Values"** layer to the top of the stack.
2. Set its **Blending Mode** to **Difference**.
3. **Result:** The image is now visualizing $Max - Min$.

---

### What to look for (The "Is it better?" Test)

Now that you have this view, compare it to your standard Saturation map (the HSB/HSL filter method we discussed earlier).

1. **Look at the Organ:**
    - **Saturation Map:** Should be Bright White.
    - **Chroma Map:** Should be Bright White (because it is high color AND high brightness).
2. **Look at the Blood Trails:**
    - **Saturation Map:** Likely Gray/White (because even though they are faint, the ratio of Red to Gray is high).
    - **Chroma Map:** Should be **Significantly Darker** (Dark Gray). Since the trails are faint, the raw distance between Max and Min is small.
3. **Look at the Paper/Background:**
    - **Both:** Should be Black.

### The Shortcut (Approximation)

If the method above is too tedious, you can get a _90% accurate_ approximation that is much faster:

1. **Duplicate** your original image.
2. **Desaturate** the bottom layer (`Ctrl+Shift+U`) to make it Grayscale ($Luminosity$).
3. Set the top (Color) layer's Blending Mode to **Difference**.
4. **Result:** This shows $|Color - Luminosity|$.
    - _Why it's different:_ It calculates distance from the _weighted average_ gray, not the mathematical center.
    - _Why it's useful:_ It is usually "good enough" to see if the blood trails drop out.

