# Pipeline Sketch and Present PET Status

> [!NOTE]  
> 
> The goal of this project is to develop classical (non–machine-learning) workflows for detecting, analyzing, and rectifying millimeter graph paper grids in ordinary laboratory photographs. Solutions must rely exclusively on deterministic computer vision, geometry, and signal-processing methods, with no neural networks.
>
> For earlier brainstorming see: [Preliminary Pipeline Notes](https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942).

## 1. High-Level Workflow

1. **Preprocessing**
    - Image enhancement
    - Illumination correction
    - Grid-targeted local contrast normalization
    - Noise management / denoising
2. **Grid Detection**
    - Segment detection (line-based)
    - Node detection (intersection-based)
3. **Raw Grid Data Filtering**
    - Cleanup, outlier removal
    - Robust centering, rotation, and scale normalization
4. **Statistical Node Set Validation**
    - Determine whether node distribution is statistically consistent with a (possibly distorted) square grid
    - See [notes](./STAT_ANALYSIS.md)
5. **Grid Data Analysis**
    - Orientation estimation
    - Pitch estimation
    - Grid-aligned bounding box detection
    - Distortion analysis
6. **Downstream Tasks**
    - Metric scale extraction (px/mm)
    - Distortion correction
    - Sample length/area measurement
    - Integration into further computational pipelines

## 2. Preprocessing

Preprocessing addresses three major objectives critical for grid extraction:
- **Illumination normalization** (remove global gradients, vignetting, shadows)
- **Local contrast enhancement** (extract grid lines even under low SNR)
- **Noise management** (prevent noise amplification during contrast boosting)

### 2.1 Illumination Correction

The preliminary preferred tool is the Retinex family (Multi-Scale [Retinex](https://imagej.net/plugins/retinex), [DI-Retinex](https://arxiv.org/abs/2404.03327), and [Fiji ImageJ Retinex](https://github.com/fiji/Fiji_Plugins/blob/main/src/main/java/Retinex_.java)). Note, [Fiji ImageJ](https://fiji.sc) Retinex is distributed as JAVA source  code which needs to be compiled with JDK for use in Fiji ImageJ; [compilation script and instructions](https://github.com/pchemguy/GeomPrimitive/tree/dev/primitives/src/pet/Fiji%20Retinex) can be obtained from Gemini / ChatGPT.

Other methods also exist (rolling-ball, large-kernel homomorphic filtering, morphological background estimation), but Retinex remains the most robust candidate for technical (non-aesthetic) illumination normalization.

Because grid detection is the primary downstream target, we must prioritize methods designed for:
- Scientific / technical imaging,
- Robustness to uneven lighting,
- Preservation of structural detail,
- Tolerance to glare and partial occlusions.

If Retinex proves sufficiently robust, implementing a Python-native version may be worthwhile.

### 2.2 Local Contrast Normalization (LCN)

Local contrast normalization is essential for improving gridline detectability when the grid is:
- faint,
- partially occluded,
- affected by plastic glare or uneven illumination.

#### OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE is a strong candidate for inclusion in the enhancement pipeline. It is locally adaptive and can improve fine structures like grid lines. The parameter-space interaction with noise amplification, illumination gradients, and node-detection success needs careful evaluation. See additional notes ([LCN](./Local Contrast Normalization) and [ref](https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942)).

#### Fiji ImageJ - Normalize Local Contrast

Currently used preprocessing:  
**Fiji ImageJ → Plugins → Integral Image Filters → Normalize Local Contrast (40×40×5.00 / center / stretch)**

This approach produces clean grid visibility and is the current baseline.

## 3. Grid Detection

Two complementary strategies are under development:
1. Segment-based detection (via LSD)
2. Node-based detection (via Sobel + intersection analysis)

These approaches can be fused for maximal robustness.

### 3.1 Segment Detection

The current prototype is implemented in **`pet_allinone.py`**, supported by the `pet_*` module family. It displays multiple debug plots and saves intermediate results (`debug_*`, `rotated_*`).

#### LSD (Line Segment Detector)

The detector uses OpenCV’s `cv2.createLineSegmentDetector`.

Important implementation notes:
- Standard pip/Conda distributions do not include advanced LSD variants (no refinement modes; limited metadata).
- Segment “width” metadata is available but less stable without opencv-contrib builds.
- A custom build of OpenCV + opencv-contrib may be necessary.

LSD returns:
- A list of detected segments
- A corresponding array of segment widths

![](./screenshots/Raw-LSD-distribution.png)
**Figure. Sample LSD Metadata Distribution**: Due to standard limited functionality, precision and NFA data is not collected. Conservative filtering may involve dropping excessively thick lines (say, top 1-5 %) and very short lines, say shorter than 2-4 pixels. Length filtering may also be attempted on bottom 1-5%, but the long tail must be kept as grid line detection may yield long segments and generally broad length distribution depending on image quality and grid size and distortions.

##### Filtering Guidelines

- Drop abnormally thick segments (top ~1-5%)
- Drop very short segments (< 2-4 px)
- Preserve broad length distribution, as real gridline detection may yield long segments.

### 3.2 Major/Minor Grid Separation

Millimeter graph paper contains:
- Major gridlines (e.g., 5 mm spacing; thicker)
- Minor gridlines (e.g., 1 mm spacing; thinner)

#### Width Distribution Analysis

Line-thickness histogram should be bimodal under reasonable image quality.

Implementation: `pet_lsd_width_analysis.py`

Major lines are typically:
- 1.5x to 3x thicker than minor lines
- More reliably detected
- Better suited as anchor geometry for early analysis

![](./screenshots/Width-Distribution-Analysis.png)
**Figure. Sample LSD Metadata Width (Line Thickness) Distribution Analysis**

![](./screenshots/Width-Splitting.png)
**Figure. Sample LSD Metadata Width (Line Thickness) Distribution Separation**

#### Orientation Analysis

Orientation distribution should also be bimodal, separated by ~90° (plus distortion).

Implementation: `pet_geom.py`

Outputs include:
- Peak detection
- Circular mean
- Circular variance
- Resultant length
- von Mises κ
- Split angle and rotation angle

This produces robust X/Y separation without any manual parameters.

![](./screenshots/Angle-KDE.png)
**Figure. Sample LSD Segment Orientation Distribution**

**Sample angle orientation analysis report:**

```
====================================================================
  ANGLE ANALYSIS REPORT
====================================================================
Angle range            : [-45.0deg, 135.0deg]
Total segments         : 1563
Total weight           : 1.000                                                                                                                                                         
Detected peaks (deg)
-------------------
  Peak 1               : -1.250
  Peak 2               : 86.500
  Split angle          : 42.625
  Rotation (deskew)    : 3.500 deg

FAMILY 1
--------
  Count                : 589
  Total weight         : 0.377
  Mean angle (deg)     :   -0.292
  Circular variance    :    0.038
  Resultant length R   :    0.962
  Kappa (von Mises)    :   13.469
  KDE bandwidth (deg)  :    5.113
  Skewness             :   -0.692
  Kurtosis             :  -21.739
  Effective N          :    589.0

FAMILY 2
--------
  Count                : 974
  Total weight         : 0.623
  Mean angle (deg)     :   85.883
  Circular variance    :    0.019
  Resultant length R   :    0.981
  Kappa (von Mises)    :   27.019
  KDE bandwidth (deg)  :    2.998
  Skewness             :    7.016
  Kurtosis             : -5396.895
  Effective N          :    974.0

====================================================================
```







##### Segment Centers

Once segments are split into major/minor and X/Y, the major X/Y families are replaced with segment centers, which are more reliable than segments themselves.

![](./screenshots/raw-lsd-segments-centers.png)
**Figure. Raw LSD Segment Centers**

![](./screenshots/major-vertical-centers.png )
**Figure. Representative LSD Segment Centers Family After Thickness and Orientation Separation** (Note, this set has also been rotated using angle obtained from angle distribution analysis)

### Node Detection

The node detector routine implemented in `pet_grid_node_detector.py` relies on Sobel operator for detecting grid line families following by intersection analysis. The routine yielded reasonable results on the tested image (for now just one), but it presently hardcodes one manually set parameter `k_len`, which is usually set around 40-70% (according to ChatGPT) of the expected pitch value. This limitation needs to be fixed, of course, replacing the hardcoded number with automatic algorithms. See preliminary [notes](./GRID_NODES_DETECTION.md) on potential strategies for automatic selection.

![](./screenshots/grid-node-detection.png)
**Figure. Grid Node Detection** (Note, the shown image also include identified grid-aligned bounding box.)

### Grid Bounding Box

A separate module implements experimental process for grid bounding box detection `pet_grid_auto_crop.py`. Presently, functionality is not integrated into main processing pipelines.

## Grid Data Analysis

### Brute-Force Black Box

Initial promising approach to solving for grid spacing has been implemented essentially as sort of black boxes (that is as [implemented by AI](https://gemini.google.com/app/1cd765eae3be9bdb))

```
pet_period.py
pet_grid_solver_extended.py
pet_grid_postprocessor.py
pet_grid_solver_xy.py
```

I am not going into further details here, as I consider an alternative approach much more promising.

### High-Level Statistical Analysis - Not Implemented

Presently not implemented at all, asking whether the obtained node set is [statistically consistent](./STAT_ANALYSIS.md) with square grids (possibly distorted), makes sense for an ML-free analysis workflow.

### Statistical Grid Pitch Estimation

> [!NOTE]
 >
 >Note, this approach was suggested by AI, and further research is necessary to verify it, as this part is beyond my expertise / general knowledge.

There is apparently a robust approach to estimating grid pitch via statistical analysis of distance distributions to nearest neighbors. A few variants have been implemented in `pet_grid_optimizer.py and `pet_grid_nodes_bbox.py`. See code for further details.

### Bounding Box Detection

With estimated pitch, there is also apparently a robust algorithm for detecting grid-aligned bounding box (see `pet_grid_nodes_bbox.py`).

### KDE-Based Marginal Density Representation

A sufficiently dense grid cloud node should have discernable grid patterns as illustrated in images above. The question is how to efficiently transform a set of node coordinates into a representation that could be used for automatic identification of these patterns without ML. A promising approach involves the following arrangement.

The 2D node pattern is projected onto horizontal axis (basically, take all x-coordinates and sort them). Next, a Gaussian-based KDE is built, which basically represents 1D (integrated over Y-coordinate) point density, and is essentially a 1D spectrum. Now, if the node cloud is rotated about its grid aligned bounding box center, the resulting KDE spectrum will evolve.

Importantly, when node cloud is not aligned with axes, they project onto X axis relatively homogenously, forming noise-only-like spectrum.

![](./screenshots/KDE-spectrum-aligned-H1-41.png)

**Figure. Real Grid Node Cloud Representations - Misaligned.** Left panel shows a conventional XY scatter plot. A large portion of the grid node is missing due to sample occlusion and plastic-file-related glares. Right panel shows half of the KDE plot. The cloud node is slightly misaligned and the associated KDE spectrum is effectively noise floor.

However, when grid lines become vertical, all nodes on those aligns project very closely (the same point for ideal grids), resulting in sharp peaks and dip valleys:

![](./screenshots/KDE-spectrum-aligned-H1-44.png)
**Figure. Real Grid Node Cloud Representations - Aligned.** Same visual as above, except the node cloud is turned by 2 deg and is aligned. KDE spectrum demonstrates typical resonant behavior.

This effect in fact has a resonant-like nature, so even moderate grid distortions can often be readily observed

![](./screenshots/KDE-spectrum-aligned-Q1.png)
**Figure. Real Grid Node Cloud Representations - Aligned - Full.** Same visual as above, except showing the full KDE spectrum on the right. While the left part of the spectrum (and left part of the node cloud) is "in focus", the left part is not.

![](./screenshots/KDE-spectrum-aligned-Q4.png)
**Figure. Real Grid Node Cloud Representations - Aligned - Q4.** Same visual as above, except the cloud is rotated by 2 deg, and the picture is opposite, with right part being in focus and left being out of focus. Note, because line intensity is directly proportional to the number of contributing points and the right part of the cloud misses considerably more points, their intensities are correspondingly weaker. But the lines are still quite sharper.

In principle, for a full 360 deg turn there are four main resonances corresponding to each grid side facing down, though the states 180 deg apart are essentially the same. For a grid with relatively few missing nodes and small distortions, there will be a number of intermediate weaker resonance corresponding alignment of nodes from different lines. The strongest of them should correspond to half a turn (45 deg for a square grid, when diagonal peak become aligned). However, diagonal alignment should be more affected by grid node grid defects. Moreover, if the present cloud is turned by 90 deg, almost all projects become severely affected by the large central defect.

Note, the distance between the sharp lines at "resonance" is the grid pitch, so we can use a variety of standard signal processing techniques to deduce the pitch. For example, with strong sharp lines, direct peak detection or 1D FFT should both be robust. Importantly both techniques can be applied to the "in-focus" portion of the spectrum only. We could also split the spectrum in several regions and apply process each at optimal angle. There is a clear physical justification for this approach, enabling us reject nosier regions of the grid before applying signal processing with solid physical justification for this approach. The important part, however, is selecting a robust numerical property sensitive to such a resonance, which could be used for automatic angle tuning.
### Automatic Tuning

There are a number of potentially suitable quantities that could be used for for present purpose, such as standard deviation / variance of KDE (contrast), which is maximized by sharp tall lines at resonance. Similarly, Shannon entropy is minimized, and Gini coefficient is maximized. Other possible candidates include signal-to-noise ratio or peak-valley difference. Importantly, all these quantities can be applied to a section of the spectrum (subset of data). For example, we can split the full region into four quartiles and treat them independently, enabling achieving optimal "local" focus, or even profiling distortion by performing angle sweep between values that focus right part and left part and tracking the focus point.

![](./screenshots/entropy-gini-sweep.png)
**Figure. Entropy ang Gini Sweep Plots.** The four panels show how Shannon entropy and the Gini coefficient change for each of the four quartiles as the node cloud performs a full turn.

![](./screenshots/entropy-stddev-sweep.png)

 **Figure. Entropy ang Gini Sweep Plots.** Same as above, except the Gini coefficient is replaced with standard deviation of KDE signal.
  