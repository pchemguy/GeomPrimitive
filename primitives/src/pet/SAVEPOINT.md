# Pipeline Sketch and Present Status

> [!NOTE]
> 
> The focus of this project is on exploring pipelines / workflows based on classic computer vision and image and signal processing algorithms not involving machine learning.  
> 
> [Preliminary pipeline notes](https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942)

## Workflow

1. Preprocessing
    - Image enhancement
    - Uneven light compensation
    - Grid-focused local contrast enhancement
    - Noise management
2. Grid detection
    - Grid segment detection
    - Grid node detection
3. Raw grid data preprocessing / cleanup / filtering
4. Statistical node data analysis (is the node set's appearance statistically comparable with square grid, [see](./STAT_ANALYSIS.md))
5. Grid data analysis
6. Downstream tasks

## Preprocessing

### Technical Photo Enhancement

An essential preprocessing objectives:
- compensating for uneven lighting / gradients / shadows
- increasing local grid contrast (managing sample contrast is a separate objective)
- managing noise (noise tends to increase with aggressive local contrast enhancement)

#### Uneven Lighting Compensation

Suggested candidate tool - [Fiji ImageJ]([https://fiji.sc](https://fiji.sc)) Retinex ([DI-Retinex](https://arxiv.org/abs/2404.03327), [Retinex](https://imagej.net/plugins/retinex), [Fiji ImageJ Retinex](https://github.com/fiji/Fiji_Plugins/blob/main/src/main/java/Retinex_.java) - note: the latter is the source code which needs to be compiled with JDK for use in Fiji ImageJ; [compilation script and instructions](https://github.com/pchemguy/GeomPrimitive/tree/dev/primitives/src/pet/Fiji%20Retinex) can be obtained from Gemini / ChatGPT).

There are other methods / algorithms / implementations designed for compensation of uneven lighting. Keep in mind that the specific downstream task - detection and analysis of millimeter graphs paper grids in non-professional ordinary lab photos with potential downstream automatic distortion compensation and/or sample area analysis with grid acting as internal scaling. For this reason, it is important to consider approaches to compensation of uneven lighting aimed for
- generic photography
- technical specialized application, where photo aesthetic quality is usually irrelevant for downstream processing tasks

Note, if Retinex proves robust, it might be worth implementing (AI-assisted) associated algos in Python.

#### Local Contrast Normalization

This processing is important. It also worth considering subsequent application of Photoshop AUTO- contrast/tone/curves/color/brightness/contrast analogs implemented in Python directly or, where available, library-based solutions. Core features of established algos / features / implementations not readily available in Python can probably be readily implemented via AI-assisted coding.

##### OpenCV - CLAHE (Contrast Limited Adaptive Histogram Equalization)

I have not carefully evaluated this feature, but it is a good candidate for integration in image enhancement pipeline (see [LCN](./Local Contrast Normalization) and [ref](https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942)).

##### Fiji ImageJ - Normalize Local Contrast

Preprocessing presently used: Fiji ImageJ ([https://fiji.sc](https://fiji.sc/)) -> Plugins -> Integral Image Filters -> Normalize Local Contrast 40x40x5.00 / center / stretch.

## Grid Detection

Presently, the project explores two independent and complementing approaches to grid detection:
- Grid segment detection (LSD)
- Grid node detection

LSD segment detection when combined with width-based (line thickness) distribution analysis for major/minor separation lines separation and angle distribution analysis for independent separation of X/Y lines yielded reasonable data (although X/Y separation may, in fact, be less important). This workflow does not involve any hardcoded manual parameters.

Sobel-based kernel edge detection following by intersection analysis yielded a comparable (LSD + width-based major/minor separation) quality data. However, present implementation involves one hardcoded tunable parameter. This manual parameter needs to be replaced with automatic selection/tuning algos.

Generally, both approaches (together with a fix for the manual parameter) should probably be combined for optimal results.

### Segment Detection

Current implementation draft is invoked by executing `pet_allinone.py`. This script depends on several other `pet_*` scripts noted below. When executed, the script will show a number of debug Matplotlib chats, as well as saves debug images in the same directory (`debug_*` and `rotated*`).

Presently, segment detection is based on OpenCV `cv2.createLineSegmentDetector` (`LSD`). Note, `Conda` and `pip` OpenCV builds do not include `opencv-contrib` features, meaning only basic LSD implementation is available (no detection refinement modes, only segment width metadata is collected). It appears that `pip` `opencv-contrib` builds are also "crippled", lacking optional more robust `LSD` variants. It might be necessary to build `opencv / opencv-contrib` from source to enable such features.

#### OpenCV LSD Segment Detection

OpenCV `cv2.createLineSegmentDetector` (`LSD`) returns a set of segment candidates ((x, y) array) and an array of associated segment width.  

![](./screenshots/Raw-LSD-distribution.png)
**Figure. Sample LSD Metadata Distribution**: Due to standard limited functionality, precision and NFA data is not collected. Conservative filtering may involve dropping excessively thick lines (say, top 1-5 %) and very short lines, say shorter than 2-4 pixels. Length filtering may also be attempted on bottom 1-5%, but the long tail must be kept as gridlines detection may very well yield long segments and generally broad length distribution depending on image quality and grid size and distortions.

#### Splitting LSD Segments into Major/Minor and X/Y

##### Major and Minor Grids 

Assuming both major and minor sub-grids are sufficiently discernable, detected segment set will include both. While both minor and major sub-grids may be potentially useful for grid analysis, initial analysis aimed at gauging major spacing and grid distortion appears to be more robust when focusing on just major grids, as minor sub-grids are thinner resulting in a substantially more sparse and irregularly appearing pattern. (I have not tried applying statistical analysis to minor sub-grid data, which might yield useful information.)

##### Width Distribution Analysis

Separating major/minor sub-grid segments is most naturally accomplished via statistical analysis of segment data. While minor segments due to potentially less reliable detection might be statistically shorter, a more direct approach is analysis of width (line thickness) metadata returned by LSD. Because major grids are conventionally thicker, sufficiently discernable grids with limited distortions should yield bimodal line thickness distribution (assuming grid segments dominate the returned data with moderate amount of noise) with two dominant peaks (major being about 1.5x to 3x thicker than minor). Core functionality related to width distribution analysis is placed in `pet_lsd_width_analysis.py`

![](./screenshots/Width-Distribution-Analysis.png)
**Figure. Sample LSD Metadata Width (Line Thickness) Distribution Analysis**

![](./screenshots/Width-Splitting.png)
**Figure. Sample LSD Metadata Width (Line Thickness) Distribution Separation**

##### Gridlines Orientation Analysis

For segment data set dominated by grid segments, segment orientation should also exhibit bimodal well-separated distribution with the two peaks roughly separated by 90 degrees (or whatever the apparent grid angle is). The core functionality related to segment orientation distribution analysis is in `pet_geom`.

![](./screenshots/Angle-KDE.png)
**Figure. Sample LSD Segment Orientation Distribution**

Sample angle orientation analysis report:

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
