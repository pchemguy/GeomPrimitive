https://gemini.google.com/app/06c7628b793a8d9e

# Automated Metrology on the Millimeter Grid: A Comprehensive Review of Bio-Image Informatics Approaches for Dimensional Analysis (2000–2025)

## Extended Abstract: Key Findings

This review synthesizes twenty-five years of bio-image informatics research (2000–2025) focused on automating the dimensional analysis of biological samples imaged on millimeter graph paper. The literature reveals a definitive shift from manual planimetry to fully automated computer vision, driven by specific algorithmic breakthroughs in noise reduction, calibration, and color segmentation.

**Key Finding 1: The Dual Utility of the Grid**

The millimeter grid functions simultaneously as a calibration lattice and a noise source. Research demonstrates that the Hough Transform is the most effective algorithm for exploiting the grid structure, enabling automated calculation of the scale factor ($px/mm$) and geometric rectification of perspective distortion, achieving measurement errors consistently below 2% in handheld mobile imaging. Conversely, for texture and venation analysis, the grid represents periodic noise effectively removed by Fast Fourier Transform (FFT) Notch Filters, which suppress grid frequencies without degrading biological feature topology.

**Key Finding 2: Superiority of Decorrelated Color Spaces**

Statistical comparisons of segmentation models highlight the inadequacy of RGB for field-based phenotyping due to high correlation between channels and sensitivity to illumination.

- **HSV (Hue-Saturation-Value):** Identified as the most robust model for mobile applications (e.g., BioLeaf). By filtering on the Saturation channel, algorithms can distinguish high-saturation biological tissues from low-saturation achromatic graph paper regardless of shadowing, improving segmentation accuracy to >99%.
- **CIELAB (Lab):** The $a^*$ channel (Green-Red axis) provides the highest statistical contrast for vegetation against soil or grid backgrounds, outperforming RGB in complex segmentation tasks with Dice Similarity Coefficients $> 0.85$.

**Key Finding 3: Evolution of Calibration Artifacts**

The necessity of the full grid has been challenged by "Red Square" calibration methods (e.g., Easy Leaf Area), which use color ratios ($G/R$) to isolate a single fiducial marker. This reduces processing time from minutes to seconds compared to full-grid analysis but introduces dependencies on physical marker placement.

**Key Finding 4: The Deep Learning Shift**

Recent applications of U-Net convolutional neural networks (2020–2025) indicate a move away from explicit grid removal. These models learn to semantically ignore grid lines as "background noise" while preserving sample integrity, offering a solution to the "dirty grid" problem that plagues classical morphological algorithms.

---

## 1. The Paradigm of the Millimeter Grid in Biological Metrology

The accurate quantification of morphological traits—morphometrics—is a foundational practice in biology, underpinning disciplines ranging from plant physiology and agronomy to entomology and forensic pathology. For over a century, the primary challenge in this field has been the conversion of qualitative visual observations into quantitative metric data. Central to this endeavor is the establishment of a reliable spatial reference system. While high-end laboratory settings may employ stereoscopic laser scanners or fixed-focal-length gantries, the vast majority of biological fieldwork and wet-lab research has relied on a ubiquitous, low-cost, and standardized fiducial marker: millimeter graph paper.

This review examines the evolution of bio-image informatics research over the last 25 years (2000–2025) concerning the processing of biological samples imaged over graph paper backgrounds. It explores the transition from manual planimetry to automated computer vision, analyzing how the geometric regularity of the grid has been exploited for calibration while simultaneously presenting a noise challenge for segmentation. A particular focus is placed on the comparative efficacy of color space transformations—specifically Hue-Saturation-Value (HSV) and CIELAB (Lab)—in facilitating robust segmentation against the challenging backdrop of grid lines and variable illumination.

### 1.1 The Historical Context: From Planimetry to Pixels

Before the advent of digital photography, the "Grid Count Method" (or Graph Paper Method) was the gold standard for measuring planar areas, such as leaf surface area or insect wing size. The protocol was deceptively simple yet laborious: a specimen was placed on a sheet of graph paper (typically with $1 \text{ mm} \times 1 \text{ mm}$ subdivisions), and its outline was traced. A researcher would then manually tally the number of full squares ($C$) and partial squares ($P$) enclosed by the outline. The total area ($A$) was derived using the formula:

$$A = (C + 0.5 \times P) \times \text{Grid Area}$$

Research by Pandey and Singh (2011) and subsequent validation studies by Radzali et al. (2016) have repeatedly confirmed that this manual method yields high accuracy, often serving as the "ground truth" against which automated systems are validated. However, the method is inherently unscalable. Processing a single large leaf with complex serrations can take 10–15 minutes, introducing operator fatigue and subjective error in estimating partial squares.

The digitization of this process began with the introduction of flatbed scanners and early digital cameras in the early 2000s. The graph paper transitioned from being a tool for manual counting to a background for digital imaging. In this new paradigm, the grid lines served a dual purpose: they provided an immediate visual scale for human observers and a potential lattice for automated calibration algorithms. However, this transition introduced the "Grid Problem": in a digital image, the grid lines represent high-frequency spatial noise that can interfere with the edge detection of biological samples, necessitating the development of robust segmentation algorithms.

### 1.2 The Physics of Calibration and Distortion

The fundamental requirement of any 2D morphometric analysis is the determination of the Scale Factor ($S_f$), typically expressed in pixels per millimeter ($px/mm$). When a sample is imaged on graph paper, the grid provides a dense field of control points that allows for sophisticated geometric corrections beyond simple linear scaling.

#### 1.2.1 Linear Scaling vs. Geometric Rectification

In ideal conditions (e.g., a flatbed scanner), the imaging plane is parallel to the sample plane, and the scale is uniform across the image. In this scenario, measuring the pixel distance between two grid lines suffices to establish $S_f$. However, in handheld photography—the dominant mode of modern fieldwork—the camera sensor is rarely perfectly parallel to the graph paper. This results in perspective distortion (keystoning) and lens distortion (barrel or pincushion effects).

Research utilizing the Hough Transform on grid paper images has demonstrated that the grid structure itself can be used to rectify these distortions. By detecting the vanishing points of the parallel grid lines, algorithms can compute a homography matrix to warp the image into an orthographic projection. This capability allows mobile applications to achieve measurement errors of less than 2% even when images are taken at oblique angles.

### 1.3 The Grid as a Segmentation Challenge

While the grid aids calibration, it hinders segmentation. In an RGB histogram, the grid lines often share intensity values with either the biological sample (e.g., dark venation on a leaf) or the background. Simple intensity thresholding (e.g., $I > 128$) often results in "grid breakthrough," where lines intersecting the sample are misclassified as holes or edges.

Consequently, the last two decades of research have bifurcated into two distinct algorithmic philosophies:

1. **Grid Removal:** Techniques employing Fast Fourier Transforms (FFT) or morphological reconstruction to erase the grid prior to analysis.
2. **Color Space Exploitation:** Techniques utilizing HSV or Lab color spaces to spectrally separate the biological pigment (e.g., chlorophyll) from the inorganic ink of the grid, rendering the lines invisible to the segmentation mask.

---

## 2. Theoretical Foundations of Image Processing on Gridded Substrates

The processing of biological images on graph paper is a composite problem involving noise reduction, feature extraction, and measurement. The literature reveals a progression from spatial domain filtering to frequency domain analysis and advanced geometric modeling.

### 2.1 Spatial Domain Analysis: Thresholding and Morphology

The most direct approach to isolating a sample from a gridded background lies in the spatial domain, manipulating pixel intensities directly.

#### 2.1.1 The LAMOS Algorithm and Otsu’s Method

A definitive example of spatial domain automation is the **LAMOS** (Leaf Area Measurement using Otsu Segmentation) method proposed by Radzali et al. (2016).16 This approach seeks to automate the manual grid count method using computer vision.

The core of LAMOS is **Otsu’s Method**, a global thresholding algorithm that assumes the image contains two classes of pixels (foreground and background) with a bi-modal histogram. Otsu’s algorithm calculates the optimal threshold $T$ that minimizes the weighted within-class variance ($\sigma_w^2$):

$$\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)$$

Where $\omega_{0,1}$ are the probabilities of the two classes and $\sigma_{0,1}^2$ are the variances. While effective for scanners, LAMOS revealed a critical limitation in camera-based images: shadowing and the similarity between dark grid lines and dark leaf veins often led to fragmentation. To counter this, the algorithm incorporates a **Median Filter** ($7 \times 7$ kernel) to smooth grid noise and **Boundary Tracing** to close gaps in the leaf edge.16

#### 2.1.2 Morphological Reconstruction for Grid Removal

When thresholding leaves artifacts—specifically grid lines that cross into the sample—**Mathematical Morphology** provides a topological solution. The morphological operations of _erosion_ and _dilation_ are foundational, but sophisticated research utilizes **Morphological Reconstruction**.

In this process, the grid lines are treated as "holes" or "thin connections." A **Morphological Closing** operation (dilation followed by erosion) using a structuring element larger than the grid line width (e.g., a disk of radius 3 pixels) can bridge the gaps caused by grid lines without significantly altering the global shape of the biological sample. For more complex grid removal, the "Rolling Ball" algorithm (a form of morphological background subtraction) effectively estimates the uneven background illumination caused by the paper's texture and removes it.

### 2.2 Frequency Domain Analysis: The FFT Approach

A more mathematically elegant solution to the "Grid Problem" exploits the periodic nature of graph paper. In the spatial domain, the grid is a complex web of edges. In the frequency domain, it simplifies to a precise geometric signature.

#### 2.2.1 Periodic Noise Suppression via Fourier Transform

When an image of graph paper is converted to the frequency domain using the **Fast Fourier Transform (FFT)**, the repetitive grid lines manifest as high-energy "spikes" or stars in the magnitude spectrum.30 The fundamental frequency of these spikes corresponds to the inverse of the grid spacing (e.g., $1 \text{ mm}^{-1}$).

Research in processing electrocardiogram (ECG) scans—which share the millimeter grid challenge with bio-imaging—has perfected the use of **Notch Filters** to remove these artifacts.4 By constructing a spectral filter that zeros out the magnitude at the specific frequencies of the grid lines (and their harmonics) while preserving the low-frequency components (the biological shape) and non-periodic high frequencies (texture), the grid can be effectively erased.

$$G(u,v) = F(u,v) \times H_{notch}(u,v)$$

Where $F(u,v)$ is the image spectrum and $H_{notch}$ is the filter mask.

This approach is particularly valuable in entomology, where researchers must analyze the delicate venation patterns of insect wings. Spatial filters might blur these veins, but FFT-based removal preserves the non-periodic vein structure while stripping the periodic background grid.

### 2.3 Geometric Modeling: The Hough Transform

While FFT removes the grid, the **Hough Transform (HT)** exploits it for calibration. The HT is a feature extraction technique used to detect parametric shapes (lines, circles) in an image.

#### 2.3.1 Automating Scale Detection

In the context of graph paper, the HT is used to detect the horizontal and vertical lines of the grid to automatically calculate the scale factor. The algorithm transforms edge points from the Cartesian space $(x,y)$ to the Hough parameter space $(\rho, \theta)$:

$$\rho = x \cos \theta + y \sin \theta$$

Collinear points in the image (a grid line) generate sinusoidal curves in Hough space that intersect at a single point $(\rho_i, \theta_i)$. By analyzing the accumulator array for peaks at $\theta \approx 0^\circ$ and $\theta \approx 90^\circ$, algorithms can identify the grid lines.

Crucially, the spacing between these peaks in the $\rho$ axis corresponds to the pixel distance between grid lines.

$$\text{Scale} (px/mm) = \text{median}(\Delta \rho_{peaks})$$

This allows software to "auto-calibrate" without user intervention. Research by Le Quang Nhat et al. (2025) and others has demonstrated that this method is robust even when the grid is partially occluded by biological samples, as the global voting mechanism of the HT is resistant to local occlusion.

---

## 3. The Chromatic Turn: Color Space Transformations in Robust Segmentation

A recurring theme in the analyzed literature is the inadequacy of the RGB color model for field-based bio-imaging. RGB channels are highly correlated; a change in ambient light intensity affects all three channels, complicating the definition of static thresholds. To address this, researchers have widely adopted decorrelated color spaces, specifically HSV and Lab.

### 3.1 The Failure of RGB in Field Conditions

In controlled laboratory environments with flatbed scanners, RGB thresholding is sufficient. However, field phenotyping apps (e.g., Leaf-IT, BioLeaf) must contend with shadows, variable cloud cover, and specular reflections. In RGB space, a "shadowed green leaf" pixel may have values closer to a "black grid line" pixel than to a "sunlit green leaf" pixel.5 This overlap in the RGB histogram renders simple segmentation impossible without complex pre-processing.

### 3.2 HSV (Hue, Saturation, Value): The "Shadow Killer"

The HSV model provides a biologically intuitive separation of chromaticity and intensity.

- **Hue (H):** Represents the color type (e.g., Green is $60^\circ–180^\circ$).
- **Saturation (S):** Represents the vibrancy of the color.
- **Value (V):** Represents brightness.

#### 3.2.1 Statistical Robustness

Research highlights HSV as the superior model for illumination invariance. By performing segmentation primarily on the **Hue** and **Saturation** channels and ignoring **Value**, algorithms become robust to shadows.

- **Case Study (BioLeaf):** The BioLeaf application utilizes HSV specifically to quantify foliar damage. The algorithm distinguishes healthy tissue (Green Hue, High Saturation) from necrotic tissue (Yellow/Brown Hue) and background holes (Low Saturation/Value). Validation against human experts showed a correlation of $R^2 = 0.94$, significantly outperforming RGB-based models ($R^2 = 0.90$) which failed under variable lighting.
- **Saturation Thresholding:** The Saturation channel is particularly effective for segmenting samples on white graph paper. The paper, being achromatic, has very low saturation ($S \approx 0$), while biological tissues (even senescent ones) typically retain higher saturation. This allows for a binary mask $M = S > T_{sat}$ that cleanly separates foreground from background regardless of lighting brightness.

### 3.3 CIELAB (Lab): Perceptual Uniformity for Precision

The CIELAB color space is designed to approximate human vision, with the $L^*$ channel for lightness and two opponent-color channels: $a^*$ (Green-Red) and $b^*$ (Blue-Yellow).

#### 3.3.1 The Power of the $a^*$ Channel

For plant phenotyping, the $a^*$ channel is a statistically powerful feature vector. The axis runs from Green (negative values) to Red (positive values).

- **Segmentation Logic:** Since plant biomass is predominantly green and soil/paper backgrounds are typically neutral or reddish-brown, a single threshold on the $a^*$ channel ($a^* < T$) often yields a near-perfect segmentation mask.
- **Performance Metrics:** Studies comparing segmentation of histological and plant samples demonstrated that Lab-based methods (specifically utilizing $a^*$) achieved higher Dice Similarity Coefficients ($DSC > 0.85$) and lower error rates compared to RGB and even HSV in scenarios with complex, textured backgrounds.
- **Execution Time vs. Accuracy:** A comparative study of color spaces for leaf segmentation found that while RGB was faster (1h 26m for a large dataset), HSV and Lab provided higher accuracy ($99.32\%$ and $97.67\%$ respectively, compared to $97.88\%$ for RGB), with HSV being the optimal trade-off between speed and precision.

### 3.4 Comparative Summary of Color Models

**Table 1: Statistical and Functional Comparison of Color Spaces in Bio-Image Segmentation**

| **Feature**                      | **RGB**                                                     | **HSV (Hue, Saturation, Value)**                                   | **CIELAB (Lab)**                                                               |
| -------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Illumination Invariance**      | **Low:** Channels are correlated; shadows alter all values. | **High:** $V$ channel isolates intensity; $H$ & $S$ remain stable. | **High:** $L^*$ isolates lightness; $a^*$ & $b^*$ represent color.             |
| **Shadow Handling**              | Poor; requires adaptive thresholding.                       | Excellent; "Shadow Killer" via $S$ and $H$ channels.               | Good; $L^*$ can be discarded, but math is complex.                             |
| **Green/Background Separation**  | Moderate; requires complex ratios (e.g., $2G-R-B$).         | High; Green is a distinct angular range in Hue.                    | **Very High:** $a^*$ axis explicitly separates Green (-) from Red/Neutral (+). |
| **Computational Cost**           | Low (Native format).                                        | Low to Moderate (Simple geometric transform).                      | Moderate (Non-linear transform requiring reference white).                     |
| **Typical Accuracy (Leaf Seg.)** | ~90–97%                                                     | **>99%**                                                           | ~97–98%                                                                        |
| **Primary Application**          | Controlled Lab Scanning (Flatbed).                          | Mobile Apps (BioLeaf), Field Phenotyping.                          | Disease Detection, Soil-Background Segmentation.                               |

---

## 4. Software Ecosystems and Algorithmic Case Studies

The implementation of these theories has resulted in a diverse ecosystem of software tools. This review categorizes them into three eras: The Desktop Era (manual/semi-auto), The "Red Square" Transition (calibration innovation), and The Mobile Field Era (fully automated).

### 4.1 The Desktop Era: ImageJ and SmartGrain

#### 4.1.1 ImageJ/Fiji: The Programmable Standard

**ImageJ** has been the workhorse of biological image analysis for 25 years.39 Its relevance to graph paper lies in its extensibility via macros.
- **Workflow:** Users typically capture an image, draw a line along the grid (e.g., spanning 10 mm), and use the `Set Scale` command to define the global calibration.
- **Automation:** Custom macros have been developed to automate "Particle Analysis." By converting images to Lab stacks and thresholding the $a^*$ channel, users can batch process folders of images. However, the dependency on manual scale setting (or consistent camera distance) remains a bottleneck for high-throughput work.

#### 4.1.2 SmartGrain: High-Throughput Seed Phenotyping

**SmartGrain**, developed by Tanabata et al. (2012), represents a specialized desktop solution for seed morphometrics.
- **Algorithmic Approach:** Unlike general-purpose tools, SmartGrain focuses on **contour analysis**. It assumes the biological samples (seeds) are disjoint objects. It employs a shape detection algorithm that minimizes the "grain-to-grain" contact issues using ellipse fitting.
- **Grid Interaction:** While capable of processing images on graph paper, SmartGrain is optimized for high-contrast backgrounds (black or white). When used with graph paper, the grid lines often require pre-processing (FFT or morphological opening) to prevent the software from interpreting grid intersections as small seeds.
- **Key Insight:** SmartGrain highlights the trade-off between specificity and generality. Its dedicated algorithms for seed shape (circularity, L/W ratio) are superior to ImageJ, but it lacks the flexibility to measure complex leaf shapes or damage.

### 4.2 The "Red Square" Revolution: Easy Leaf Area

A significant leap in automation came with **Easy Leaf Area** (Easlon & Bloom, 2014), which challenged the necessity of the graph paper grid itself.
- **Calibration Innovation:** Instead of analyzing a complex grid, the software uses a **red calibration square** of known area (typically $4 \text{ cm}^2$) placed in the same plane as the leaf.
- **Algorithm:** The software avoids complex color spaces, relying instead on computationally cheap RGB ratios:
    - **Leaf:** $\text{Green} > \text{Red}$ AND $\text{Green} > \text{Blue}$
    - **Scale:** $\text{Red} > \text{Green}$
- Calculation:

$$A_{leaf} = \frac{\text{Count}_{green}}{\text{Count}_{red}} \times A_{scale}$$    
- **Performance:** Validation studies show that Easy Leaf Area matches the accuracy of the manual graph paper method ($R^2 > 0.99$) but reduces processing time from minutes to seconds.10 Its open-source Python codebase allowed it to be rapidly adopted and modified for various crop types.
- **Critique:** While effective, it requires the physical preparation of the red square. If the square is tilted or partially occluded, the calibration fails.

### 4.3 The Mobile Field Era: BioLeaf, Leaf-IT, and Petiole

The current state-of-the-art leverages smartphone sensors and processors to perform analysis in situ.

#### 4.3.1 Leaf-IT: Solving Margin Complexity

**Leaf-IT** (Schrader et al., 2017) addresses the limitations of thresholding for leaves with complex margins (e.g., serrated or deeply lobed leaves).

- **Robust Margin Detection:** Instead of simple binarization, Leaf-IT employs an edge-detection algorithm that is robust to "impurities" (shadows, dirt).
- **Calibration:** It offers two modes: "Set Size" (user defines a length) and "Reference Object" (user defines an area).
- **Limitation:** Research notes that while highly accurate for simple shapes, its margin detection can struggle with damaged leaves or very complex morphologies where the "inside" vs. "outside" distinction is ambiguous.

#### 4.3.2 BioLeaf: Quantifying Defoliation

**BioLeaf** (Machado et al., 2016) is a specialized tool for calculating leaf damage (herbivory).

- **Algorithmic Innovation:** It not only segments the leaf using HSV but also **reconstructs** the original leaf shape. Using morphological operations to close small holes and **Bézier curves** to interpolate large missing bite marks, it estimates the _potential_ area vs. the _actual_ area.
- **Validation:** The app demonstrated precision comparable to human specialists, democratizing the collection of herbivory data which previously required subjective estimation scales.

#### 4.3.3 Petiole: The Standardized Pad

**Petiole** (and Petiole Pro) represents the commercial maturation of this technology.
- **Calibration Pads:** Moving beyond graph paper, Petiole uses specific checkerboard calibration pads. The app uses corner detection (likely Harris Corner Detector or similar) to identify the checkerboard, automatically calculating scale and correcting for camera tilt.
- **Accuracy:** Independent studies comparing Petiole to the manual grid count method report errors $< 1 \text{ cm}^2$ and high consistency, validating the shift from manual counting to app-based measurement.

---

## 5. Domain-Specific Applications and Challenges

### 5.1 Entomology: Wing Morphometrics on the Grid

In entomology, the measurement of insect wing venation is critical for taxonomy.
- **The Grid as Background:** Graph paper is often used as a standard background for pinning specimens.
- **Automated vs. Manual:** While tools like **AutoCAD** have been used to manually digitize landmarks from scanned graph paper images, modern tools like **DrawWing** and **DeepWings** automate this.
- **Challenge:** The grid lines can be confused with wing veins.
- **Solution:** This is a primary use case for **FFT Notch Filtering**. By removing the periodic grid noise, the non-periodic wing veins become clear for automated topology extraction.
- **Significance:** Automated geometric morphometrics on graph paper has enabled large-scale population monitoring (e.g., bees) to assess the impact of climate change on body size.

### 5.2 Forensics and Medical Imaging

The intersection of biology and forensics relies heavily on scaled photography.
- **Skin Lesion Analysis:** Dermatologists measure lesions to track malignancy (melanoma). Research shows that **Lab color space** is crucial here for segmenting the "reddish" lesion from "pinkish" skin, a contrast that RGB often fails to capture. The graph paper (or ruler) provides the critical metric of _growth rate_.
- **ECG Digitization:** While not "wet" biology, the digitization of ECG paper strips is technically identical to the biological grid problem. Deep Learning models (U-Net) and Hough Transforms developed for ECGs are now being cross-applied to biological phenotyping to remove grid lines from archival data.

### 5.3 Deep Learning and the Future of the Grid

The most recent research (2020–2025) explores using Convolutional Neural Networks (CNNs) to bypass the "segmentation" phase entirely.
- **U-Net on Millimeter Paper:** Studies have trained U-Net architectures on synthetic datasets of leaves placed on virtual graph paper.
- **Mechanism:** The network learns to ignore the grid lines implicitly, recognizing the "leaf" texture features.
- **Implication:** This suggests a future where explicit grid removal (FFT/Morphology) is obsolete. The AI simply "sees through" the grid, using it only for scale inference (via a parallel regression head) while segmenting the sample based on learned semantic features.

---

## 6. Unsatisfied Requirements and Missing Information

While the reviewed literature is extensive, several gaps and unsatisfied requirements in the current technology base were identified:

1. **Universal "Dirty" Grid Handling:** Most algorithms (SmartGrain, LAMOS) perform optimally on clean, pristine graph paper. There is a lack of robust solutions for "dirty" field conditions where the paper is crumpled, stained with soil, or wet. Current methods (Leaf-IT) attempt to mitigate this, but valid "in-the-wild" robustness remains a challenge that often forces researchers back to manual counting.
2. **3D Structure Calibration:** Current methods assume the leaf is a 2D plane parallel to the calibration grid. For curved, cup-shaped, or thick samples (e.g., succulents, fruits), the single-plane calibration introduces significant projection errors. There is a lack of accessible tools that utilize "Structure from Motion" (SfM) or depth-sensing (LiDAR on newer iPhones) to correct for sample volume in bio-morphometrics.
3. **Standardized Color Calibration:** While apps use HSV/Lab for segmentation, they rarely calibrate the _sensor's_ color response. Differences in camera firmware (Samsung vs. iPhone) can shift the Hue values, potentially affecting threshold-based segmentation (e.g., BioLeaf's damage detection). True color calibration using standard color charts (Macbeth charts) is largely absent in these low-cost apps.

---

## 7. Conclusion

The transition from the manual **Grid Count Method** to automated **Bio-Image Informatics** represents a triumph of applied computer vision. The millimeter graph paper, once a tool for tedious manual integration, has evolved into a sophisticated fiducial marker that anchors digital biological data to the physical world.

This review has demonstrated that:
1. **Color Space Matters:** The shift from RGB to **HSV** and **Lab** is not merely technical but essential for field robustness, offering statistical improvements in segmentation accuracy from ~90% to >99%.
2. **Algorithms are Specialized:** There is no "one size fits all." **Hough Transforms** excel at scale detection, **FFT** excels at grid removal for texture analysis, and **Morphological Reconstruction** excels at shape preservation.
3. **The Future is Hybrid:** The next generation of tools will likely combine the reliability of physical markers (grids/pads) with the semantic understanding of Deep Learning (U-Nets), finally solving the "dirty grid" and "complex margin" problems that remain the last barriers to fully automated, high-throughput biological metrology.

### **Table 2: Evolution of Biological Dimensioning Methods (2000–2025)**

|**Method**|**Era**|**Primary Algorithm**|**Grid Role**|**Key Advantage**|**Key Limitation**|
|---|---|---|---|---|---|
|**Grid Count**|Pre-2000s|Manual Counting|Measurement Unit|Gold Standard Accuracy|Extremely Slow, Subjective|
|**ImageJ Macros**|2000–2010|Thresholding (RGB)|Scale Reference|Digital Archiving|Manual Scale Setting, Lighting Sensitive|
|**SmartGrain**|2010–2015|Contour Analysis|Background (Ignored)|High Throughput (Seeds)|Requires Clean Background|
|**Easy Leaf Area**|2014|Color Ratios (G/R)|Replaced by Red Square|Auto-Calibration, Speed|Requires Specific Red Marker|
|**BioLeaf / Leaf-IT**|2015–Present|HSV / Margin Detection|Background / Scale|Field Robustness (Shadows)|Margin errors on complex shapes|
|**Deep Learning**|2020–Present|U-Net / CNNs|Implicit Context|"Sees Through" Grid|Requires Large Training Data|

## References

1. Liquid flow in scaffold derived from natural source: experimental observations and biological outcome | Regenerative Biomaterials | Oxford Academic, accessed December 4, 2025, [https://academic.oup.com/rb/article/doi/10.1093/rb/rbac034/6595038](https://academic.oup.com/rb/article/doi/10.1093/rb/rbac034/6595038)  
2. Analysing Arbitrary Curves from the Line Hough Transform \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/2313-433X/6/4/26](https://www.mdpi.com/2313-433X/6/4/26)  
3. Advanced Leaf Vein Pattern Extraction and Analysis using Machine Learning Algorithms, accessed December 4, 2025, [https://www.researchgate.net/publication/391321707\_Advanced\_Leaf\_Vein\_Pattern\_Extraction\_and\_Analysis\_using\_Machine\_Learning\_Algorithms](https://www.researchgate.net/publication/391321707_Advanced_Leaf_Vein_Pattern_Extraction_and_Analysis_using_Machine_Learning_Algorithms)  
4. Deep Learning-Based Digitization of Overlapping ECG Images with Open-Source Python Code \- arXiv, accessed December 4, 2025, [https://arxiv.org/html/2506.10617v1](https://arxiv.org/html/2506.10617v1)  
5. Compressive Sensing of Medical Images Based on HSV Color Space \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/1424-8220/23/5/2616](https://www.mdpi.com/1424-8220/23/5/2616)  
6. Estimating soybean leaf defoliation using convolutional neural networks and synthetic images | Request PDF \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/330058418\_Estimating\_soybean\_leaf\_defoliation\_using\_convolutional\_neural\_networks\_and\_synthetic\_images](https://www.researchgate.net/publication/330058418_Estimating_soybean_leaf_defoliation_using_convolutional_neural_networks_and_synthetic_images)  
7. Spatial statistics for segmenting histological structures in H\&E stained tissue images \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5498226/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5498226/)  
8. Region Adjacency Graph Approach for Acral Melanocytic Lesion Segmentation \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/2076-3417/8/9/1430](https://www.mdpi.com/2076-3417/8/9/1430)  
9. accessed December 4, 2025, [https://www.researchgate.net/publication/264081837\_Easy\_Leaf\_Area\_Automated\_Digital\_Image\_Analysis\_for\_Rapid\_and\_Accurate\_Measurement\_of\_Leaf\_Area\#:\~:text=Methods%20and%20Results%3A%20Easy%20Leaf,measurement%20that%20other%20software%20methods](https://www.researchgate.net/publication/264081837_Easy_Leaf_Area_Automated_Digital_Image_Analysis_for_Rapid_and_Accurate_Measurement_of_Leaf_Area#:~:text=Methods%20and%20Results%3A%20Easy%20Leaf,measurement%20that%20other%20software%20methods)  
10. A Novel Method for Leaf Area Estimation based on Hough Transform \- dline.info, accessed December 4, 2025, [https://www.dline.info/jmpt/fulltext/v9n2/jmptv9n2\_1.pdf](https://www.dline.info/jmpt/fulltext/v9n2/jmptv9n2_1.pdf)  
11. Predicting yield of individual field-grown rapeseed plants from rosette-stage leaf gene expression | PLOS Computational Biology \- Research journals, accessed December 4, 2025, [https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011161](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011161)  
12. LAESI: Leaf Area Estimation with Synthetic Imagery \- arXiv, accessed December 4, 2025, [https://arxiv.org/html/2404.00593v1](https://arxiv.org/html/2404.00593v1)  
13. Measuring Leaf Area \- K-State Agronomy, accessed December 4, 2025, [https://www.agronomy.k-state.edu/outreach-and-services/educational-workshops/willie-and-the-beanstalk/leaf\_area.html](https://www.agronomy.k-state.edu/outreach-and-services/educational-workshops/willie-and-the-beanstalk/leaf_area.html)  
14. Easy Leaf Area: Automated Digital Image Analysis for Rapid and Accurate Measurement of Leaf Area \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/264081837\_Easy\_Leaf\_Area\_Automated\_Digital\_Image\_Analysis\_for\_Rapid\_and\_Accurate\_Measurement\_of\_Leaf\_Area](https://www.researchgate.net/publication/264081837_Easy_Leaf_Area_Automated_Digital_Image_Analysis_for_Rapid_and_Accurate_Measurement_of_Leaf_Area)  
15. Performance of the petiole mobile application on the leaf area estimation as varied with calibration height \- The Pharma Innovation Journal, accessed December 4, 2025, [https://www.thepharmajournal.com/archives/2021/vol10issue4S/PartF/S-10-4-71-974.pdf](https://www.thepharmajournal.com/archives/2021/vol10issue4S/PartF/S-10-4-71-974.pdf)  
16. Measuring Leaf Area using Otsu Segmentation Method (LAMOS) \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/313418145\_Measuring\_Leaf\_Area\_using\_Otsu\_Segmentation\_Method\_LAMOS](https://www.researchgate.net/publication/313418145_Measuring_Leaf_Area_using_Otsu_Segmentation_Method_LAMOS)  
17. Measuring Leaf Area using Otsu Segmentation Method (LAMOS), accessed December 4, 2025, [https://sciresol.s3.us-east-2.amazonaws.com/IJST/Articles/2016/Issue-48/Article141.pdf](https://sciresol.s3.us-east-2.amazonaws.com/IJST/Articles/2016/Issue-48/Article141.pdf)  
18. Estimation of leaf area by mobile application: Fast and accurate method \- The Pharma Innovation Journal, accessed December 4, 2025, [https://www.thepharmajournal.com/archives/2021/vol10issue4S/PartE/S-10-4-38-448.pdf](https://www.thepharmajournal.com/archives/2021/vol10issue4S/PartE/S-10-4-38-448.pdf)  
19. Removing grid paper lines : r/GIMP \- Reddit, accessed December 4, 2025, [https://www.reddit.com/r/GIMP/comments/pdx0se/removing\_grid\_paper\_lines/](https://www.reddit.com/r/GIMP/comments/pdx0se/removing_grid_paper_lines/)  
20. An Efficient Computational Framework for the Analysis of Whole Slide Images: Application to Follicular Lymphoma Immunohistochemistry \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3432990/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3432990/)  
21. ± Distortion of an endoscopic image of millimetre-grid graph paper... | Download Scientific Diagram \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/figure/Distortion-of-an-endoscopic-image-of-millimetre-grid-graph-paper-obtained-using-a-BF-1\_fig1\_227699659](https://www.researchgate.net/figure/Distortion-of-an-endoscopic-image-of-millimetre-grid-graph-paper-obtained-using-a-BF-1_fig1_227699659)  
22. Estimation of leaf area by mobile application: Fast and accurate method \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/351031561\_Estimation\_of\_leaf\_area\_by\_mobile\_application\_Fast\_and\_accurate\_method](https://www.researchgate.net/publication/351031561_Estimation_of_leaf_area_by_mobile_application_Fast_and_accurate_method)  
23. Visualizing Plant Responses: Novel Insights Possible Through Affordable Imaging Techniques in the Greenhouse \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11511021/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11511021/)  
24. Fast and Accurate Method for Leaf Area Measurement \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/266058653\_Fast\_and\_Accurate\_Method\_for\_Leaf\_Area\_Measurement](https://www.researchgate.net/publication/266058653_Fast_and_Accurate_Method_for_Leaf_Area_Measurement)  
25. Acquisition of a single grid-based phase-contrast X-ray image using instantaneous frequency and noise filtering \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9793636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9793636/)  
26. BioLeaf: A professional mobile application to measure foliar damage caused by insect herbivory | Request PDF \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/308537237\_BioLeaf\_A\_professional\_mobile\_application\_to\_measure\_foliar\_damage\_caused\_by\_insect\_herbivory](https://www.researchgate.net/publication/308537237_BioLeaf_A_professional_mobile_application_to_measure_foliar_damage_caused_by_insect_herbivory)  
27. CellProfiler \- AWS, accessed December 4, 2025, [https://cellprofiler-manual.s3.amazonaws.com/cp2\_manual\_9978.pdf](https://cellprofiler-manual.s3.amazonaws.com/cp2_manual_9978.pdf)  
28. Automating the characterisation of beach microplastics through the application of image analyses \- Archimer, accessed December 4, 2025, [https://archimer.ifremer.fr/doc/00513/62471/69557.pdf](https://archimer.ifremer.fr/doc/00513/62471/69557.pdf)  
29. Development and Validation of an Algorithm for the Digitization of ECG Paper Images \- PMC, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9572306/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9572306/)  
30. Extract text from background grids/lines \[2\] \- Stack Overflow, accessed December 4, 2025, [https://stackoverflow.com/questions/72752475/extract-text-from-background-grids-lines-2](https://stackoverflow.com/questions/72752475/extract-text-from-background-grids-lines-2)  
31. Issues and research on foetal electrocardiogram signal elicitation | Request PDF, accessed December 4, 2025, [https://www.researchgate.net/publication/259524322\_Issues\_and\_research\_on\_foetal\_electrocardiogram\_signal\_elicitation](https://www.researchgate.net/publication/259524322_Issues_and_research_on_foetal_electrocardiogram_signal_elicitation)  
32. (PDF) WingAnalogy: a computer vision-based tool for automated insect wing asymmetry and morphometry analysis \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/384407878\_WingAnalogy\_a\_computer\_vision-based\_tool\_for\_automated\_insect\_wing\_asymmetry\_and\_morphometry\_analysis](https://www.researchgate.net/publication/384407878_WingAnalogy_a_computer_vision-based_tool_for_automated_insect_wing_asymmetry_and_morphometry_analysis)  
33. Automated Grid Detection in Hemocytometer Images Using Canny and Hough Line Transform for Cell Quantification | Request PDF \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/395484582\_Automated\_Grid\_Detection\_in\_Hemocytometer\_Images\_Using\_Canny\_and\_Hough\_Line\_Transform\_for\_Cell\_Quantification](https://www.researchgate.net/publication/395484582_Automated_Grid_Detection_in_Hemocytometer_Images_Using_Canny_and_Hough_Line_Transform_for_Cell_Quantification)  
34. Why do we use the HSV colour space so often in vision and image processing?, accessed December 4, 2025, [https://dsp.stackexchange.com/questions/2687/why-do-we-use-the-hsv-colour-space-so-often-in-vision-and-image-processing](https://dsp.stackexchange.com/questions/2687/why-do-we-use-the-hsv-colour-space-so-often-in-vision-and-image-processing)  
35. A New Method for Segmentation of Images Represented in a HSV Color Space, accessed December 4, 2025, [https://www.researchgate.net/publication/220785137\_A\_New\_Method\_for\_Segmentation\_of\_Images\_Represented\_in\_a\_HSV\_Color\_Space](https://www.researchgate.net/publication/220785137_A_New_Method_for_Segmentation_of_Images_Represented_in_a_HSV_Color_Space)  
36. A Semiautomatic Multi-Label Color Image Segmentation Coupling Dirichlet Problem and Colour Distances \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8539020/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8539020/)  
37. CropGCNN: color space-based crop disease classification using group convolutional neural network \- PMC \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11322995/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322995/)  
38. Color disease leaf image segmentation using NAMS superpixel algorithm \- PMC \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6004959/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6004959/)  
39. Automated Quantification and Analysis of Cell Counting Procedures Using ImageJ Plugins, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5226253/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5226253/)  
40. 4 Proven Methods of How to Measure Leaf Area \- Petiole Pro, accessed December 4, 2025, [https://www.petiolepro.com/blog/4-proven-methods-of-how-to-measure-leaf-area/](https://www.petiolepro.com/blog/4-proven-methods-of-how-to-measure-leaf-area/)  
41. Two-photon volumetric study of cleared, invasive ductal carcinoma breast tissue samples and associated axillary lymph nodes \- bioRxiv, accessed December 4, 2025, [https://www.biorxiv.org/content/10.1101/2025.11.07.687157v1.full.pdf](https://www.biorxiv.org/content/10.1101/2025.11.07.687157v1.full.pdf)  
42. Study of Mechanical Response of Paper-Based Microfluidic System as a Potential Milk Tester \- PMC \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10386323/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10386323/)  
43. r-link/leaf\_area\_ImageJ: A simple tutorial for the estimation of leaf area with ImageJ (for in-class use) \- GitHub, accessed December 4, 2025, [https://github.com/r-link/leaf\_area\_ImageJ](https://github.com/r-link/leaf_area_ImageJ)  
44. SmartGrain: High-Throughput Phenotyping Software for Measuring Seed Shape through Image Analysis \- PMC \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3510117/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3510117/)  
45. SmartGrain automatically identifies seeds within an image and measures... \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/figure/SmartGrain-automatically-identifies-seeds-within-an-image-and-measures-their-shape\_fig1\_232231204](https://www.researchgate.net/figure/SmartGrain-automatically-identifies-seeds-within-an-image-and-measures-their-shape_fig1_232231204)  
46. GridFree: a python package of imageanalysis for interactive grain counting and measuring | Plant Physiology | Oxford Academic, accessed December 4, 2025, [https://academic.oup.com/plphys/article/186/4/2239/6274909](https://academic.oup.com/plphys/article/186/4/2239/6274909)  
47. SmartGrain: High-Throughput Phenotyping Software for Measuring Seed Shape through Image Analysis1\[C\]\[W\]\[OA\] \- Semantic Scholar, accessed December 4, 2025, [https://www.semanticscholar.org/paper/SmartGrain%3A-High-Throughput-Phenotyping-Software-Tanabata-Shibaya/8e0d5bf8759170beff845553cde84a056bbced43](https://www.semanticscholar.org/paper/SmartGrain%3A-High-Throughput-Phenotyping-Software-Tanabata-Shibaya/8e0d5bf8759170beff845553cde84a056bbced43)  
48. Automated digital image analysis for rapid and accurate measurement of leaf area \- PMC \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4103476/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4103476/)  
49. Accelerating leaf area measurement using a volumetric approach \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/352991661\_Accelerating\_leaf\_area\_measurement\_using\_a\_volumetric\_approach](https://www.researchgate.net/publication/352991661_Accelerating_leaf_area_measurement_using_a_volumetric_approach)  
50. (PDF) Leaf-IT: An Android application for measuring leaf area \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/320466650\_Leaf-IT\_An\_Android\_application\_for\_measuring\_leaf\_area](https://www.researchgate.net/publication/320466650_Leaf-IT_An_Android_application_for_measuring_leaf_area)  
51. A professional mobile application to measure foliar damage caused by insect herbivory \- BioLeaf, accessed December 4, 2025, [https://bioleaf.icmc.usp.br/paper/2016BIOLEAF.pdf](https://bioleaf.icmc.usp.br/paper/2016BIOLEAF.pdf)  
52. PetiolePro™ — Leaf Area Made Easy, accessed December 4, 2025, [https://www.petiolepro.com/leaf-area-meter-petiole-pro/](https://www.petiolepro.com/leaf-area-meter-petiole-pro/)  
53. (PDF) Performance of the petiole mobile application on the leaf area estimation as varied with calibration height \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/355449012\_Performance\_of\_the\_petiole\_mobile\_application\_on\_the\_leaf\_area\_estimation\_as\_varied\_with\_calibration\_height](https://www.researchgate.net/publication/355449012_Performance_of_the_petiole_mobile_application_on_the_leaf_area_estimation_as_varied_with_calibration_height)  
54. Automatic Measurement of the Morphological Characteristics Of Honeybees With A Computational Program | Request PDF \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/362932887\_Automatic\_Measurement\_of\_the\_Morphological\_Characteristics\_Of\_Honeybees\_With\_A\_Computational\_Program](https://www.researchgate.net/publication/362932887_Automatic_Measurement_of_the_Morphological_Characteristics_Of_Honeybees_With_A_Computational_Program)  
55. (PDF) Chapter 15: Automatic measurement of the morphological characteristics of honeybees with a computational program, Book Chapter from “Who runs the world: data” \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/348000140\_Chapter\_15\_Automatic\_measurement\_of\_the\_morphological\_characteristics\_of\_honeybees\_with\_a\_computational\_program\_Book\_Chapter\_from\_Who\_runs\_the\_world\_data](https://www.researchgate.net/publication/348000140_Chapter_15_Automatic_measurement_of_the_morphological_characteristics_of_honeybees_with_a_computational_program_Book_Chapter_from_Who_runs_the_world_data)  
56. Stayin' alive: Optimizing wing geometric morphometrics toward a harmless method | Request PDF \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/393491797\_Stayin'\_alive\_Optimizing\_wing\_geometric\_morphometrics\_toward\_a\_harmless\_method](https://www.researchgate.net/publication/393491797_Stayin'_alive_Optimizing_wing_geometric_morphometrics_toward_a_harmless_method)  
57. CHAPTER 15 AUTOMATIC MEASUREMENT OF THE MORPHOLOGICAL CHARACTERISTICS OF HONEYBEES WITH A COMPUTATIONAL PROGRAM, accessed December 4, 2025, [https://cdn.istanbul.edu.tr/file/JTA6CLJ8T5/2F7042B41E154D35B4448BF9D3B68DEC](https://cdn.istanbul.edu.tr/file/JTA6CLJ8T5/2F7042B41E154D35B4448BF9D3B68DEC)  
58. An Efficient Algorithm for Automated Skin Lesion Detection: A Non-Machine Learning Approach \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/384675205\_An\_Efficient\_Algorithm\_for\_Automated\_Skin\_Lesion\_Detection\_A\_Non-Machine\_Learning\_Approach](https://www.researchgate.net/publication/384675205_An_Efficient_Algorithm_for_Automated_Skin_Lesion_Detection_A_Non-Machine_Learning_Approach)  
59. Cardiovascular Diseases Prediction From ECG Images | PDF | Electrocardiography \- Scribd, accessed December 4, 2025, [https://www.scribd.com/document/861430810/ppt-Cardiovascular-Diseases-Prediction-from-ECG-images](https://www.scribd.com/document/861430810/ppt-Cardiovascular-Diseases-Prediction-from-ECG-images)  
60. Automatic Measurement of Seed Geometric Parameters Using a Handheld Scanner \- PMC, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11436011/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11436011/)  
61. Leaf‐IT: An Android application for measuring leaf area \- PMC \- NIH, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5696424/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5696424/)  
62. LAESI: Leaf Area Estimation with Synthetic Imagery \- Algorithmic Botany, accessed December 4, 2025, [https://algorithmicbotany.org/papers/syntheticleaves2024.cvpr.pdf](https://algorithmicbotany.org/papers/syntheticleaves2024.cvpr.pdf)  
63. Deep learning‐ and image processing‐based methods for automatic estimation of leaf herbivore damage \- Archipel UQAM, accessed December 4, 2025, [https://archipel.uqam.ca/17463/1/Wang%20et%20al%202024.pdf](https://archipel.uqam.ca/17463/1/Wang%20et%20al%202024.pdf)  
64. Developing a New Method of Transformation for Obtaining XYZ Color Values from RGB Images for Agricultural Applications \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/1424-8220/24/23/7728](https://www.mdpi.com/1424-8220/24/23/7728)  
65. A Three-Dimensional Hough Transform-Based Track-Before-Detect Technique for Detecting Extended Targets in Strong Clutter Backgrounds \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/1424-8220/19/4/881](https://www.mdpi.com/1424-8220/19/4/881)  
66. leaf area duration: Topics by Science.gov, accessed December 4, 2025, [https://www.science.gov/topicpages/l/leaf+area+duration](https://www.science.gov/topicpages/l/leaf+area+duration)  
67. Leaf Area Estimation by Photographing Leaves Sandwiched between Transparent Clear File Folder Sheets \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/2311-7524/9/6/709](https://www.mdpi.com/2311-7524/9/6/709)  
68. A fully automatic gridding method for cDNA microarray images \- PMC \- PubMed Central, accessed December 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3110145/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3110145/)  
69. Grid Screener: A Tool for Automated High-throughput Screening on Biochemical and Biological Analysis Platforms \- Publikationen in KITopen, accessed December 4, 2025, [https://publikationen.bibliothek.kit.edu/1000141260](https://publikationen.bibliothek.kit.edu/1000141260)  
70. A Novel Method for ECG Paper Records Digitization \- Computing in Cardiology, accessed December 4, 2025, [https://www.cinc.org/2019/Program/accepted/264\_CinCFinalPDF.pdf](https://www.cinc.org/2019/Program/accepted/264_CinCFinalPDF.pdf)  
71. Deep Learning-Based Methods for Automated Estimation of Insect Length, Volume, and Biomass \- bioRxiv, accessed December 4, 2025, [https://www.biorxiv.org/content/biorxiv/early/2025/05/27/2025.05.22.655251.full.pdf](https://www.biorxiv.org/content/biorxiv/early/2025/05/27/2025.05.22.655251.full.pdf)  
72. Using the Software DeepWings© to Classify Honey Bees across Europe through Wing Geometric Morphometrics \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/2075-4450/13/12/1132](https://www.mdpi.com/2075-4450/13/12/1132)  
73. periodic noise detection in image's frequency domain \- Signal Processing Stack Exchange, accessed December 4, 2025, [https://dsp.stackexchange.com/questions/75035/periodic-noise-detection-in-images-frequency-domain](https://dsp.stackexchange.com/questions/75035/periodic-noise-detection-in-images-frequency-domain)  
74. Data & Methods – Plant Ecology Lab – Macquarie University, accessed December 4, 2025, [https://julianschrader.wordpress.com/data/](https://julianschrader.wordpress.com/data/)  
75. Deep learning‐ and image processing‐based methods for automatic estimation of leaf herbivore damage | Request PDF \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/378574386\_Deep\_learning-\_and\_image\_processing-based\_methods\_for\_automatic\_estimation\_of\_leaf\_herbivore\_damage](https://www.researchgate.net/publication/378574386_Deep_learning-_and_image_processing-based_methods_for_automatic_estimation_of_leaf_herbivore_damage)  
76. Automated Large-Scale Tornado Treefall Detection and Directional Analysis Using Machine Learning in \- AMS Journals, accessed December 4, 2025, [https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0062.1.xml](https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0062.1.xml)  
77. Sonar-based localization of mobile robots using the Hough transform \- Calhoun, accessed December 4, 2025, [https://calhoun.nps.edu/server/api/core/bitstreams/6ba25f0b-1aba-44f5-9dff-d897e12be7f9/content](https://calhoun.nps.edu/server/api/core/bitstreams/6ba25f0b-1aba-44f5-9dff-d897e12be7f9/content)  
78. An Improved Simple Morphological Filter for the Terrain Classification of Airborne LIDAR Data \- ResearchGate, accessed December 4, 2025, [https://www.researchgate.net/publication/258333806\_An\_Improved\_Simple\_Morphological\_Filter\_for\_the\_Terrain\_Classification\_of\_Airborne\_LIDAR\_Data](https://www.researchgate.net/publication/258333806_An_Improved_Simple_Morphological_Filter_for_the_Terrain_Classification_of_Airborne_LIDAR_Data)  
79. Image Processing and Optimization for Higher Order Aberration Correction and Power Distribution Management \- Scholarship@Miami, accessed December 4, 2025, [https://scholarship.miami.edu/view/pdfCoverPage?instCode=01UOML\_INST\&filePid=13378599540002976\&download=true](https://scholarship.miami.edu/view/pdfCoverPage?instCode=01UOML_INST&filePid=13378599540002976&download=true)  
80. Straight Line Detection with the Hough Transform Method based on a Rectangular Grid, accessed December 4, 2025, [https://www.researchgate.net/publication/344726289\_Straight\_Line\_Detection\_with\_the\_Hough\_Transform\_Method\_based\_on\_a\_Rectangular\_Grid](https://www.researchgate.net/publication/344726289_Straight_Line_Detection_with_the_Hough_Transform_Method_based_on_a_Rectangular_Grid)  
81. Machine Learning-Powered Segmentation of Forage Crops in RGB Imagery Through Artificial Sward Images \- MDPI, accessed December 4, 2025, [https://www.mdpi.com/2073-4395/15/2/356](https://www.mdpi.com/2073-4395/15/2/356)  
82. Predicting yield traits of individual field-grown Brassica napus plants from rosette-stage leaf gene expression \- bioRxiv, accessed December 4, 2025, [http://biorxiv.org/cgi/reprint/2022.10.21.513275v1](http://biorxiv.org/cgi/reprint/2022.10.21.513275v1)