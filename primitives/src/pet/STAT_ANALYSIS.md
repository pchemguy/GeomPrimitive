# Stat Analysis of Grid Node Set

## Preliminary Statistical Assessment of Node Set as Belonging to a Distorted Square Grid

**REF**: https://aistudio.google.com/prompts/1F-lB6imZS6Rr2YeUuSt-4s-ehLAqAbas

To assess whether a set of points belongs to a distorted square grid, you need statistics that measure **local order** (do neighbors look like a square?), **periodicity** (is there a repeating pattern?), and **topology** (connectivity).

Here is a breakdown of the statistics and methods you can use, ranging from simple geometric checks to advanced order parameters.

---

### 1. Local Geometric Statistics
These methods look at the immediate relationship between a point and its neighbors. They are best for detecting **non-linear distortions** (warping) where the grid structure is locally preserved but globally bent.

#### A. The Bond Orientational Order Parameter ($\psi_4$)
This is the gold standard in condensed matter physics for distinguishing between square, hexagonal, and liquid (random) phases.
*   **The Statistic:** For every point $j$, identify its $k$ nearest neighbors (usually 4). Calculate the angle $\theta_{jk}$ of the bond between point $j$ and neighbor $k$ relative to a fixed axis.
    $$ \psi_4 = \left| \frac{1}{N} \sum_{j=1}^{N} \left( \frac{1}{4} \sum_{k=1}^{4} e^{4i\theta_{jk}} \right) \right| $$
*   **Interpretation:**
    *   $\psi_4 \approx 1$: Perfect square grid (even if rotated).
    *   $\psi_4 \approx 0$: Random arrangement or hexagonal packing.
    *   **Note:** If the grid is locally square but the orientation rotates gradually across the domain (warping), calculate the magnitude per point before averaging, or look at the local variance.

#### B. Nearest Neighbor Distance Consistency
*   **The Statistic:** Calculate the distance from every point to its 1st nearest neighbor ($d_1$). Compute the **Coefficient of Variation (CV)** of these distances:
    $$ CV = \frac{\sigma_{d_1}}{\mu_{d_1}} $$
*   **Interpretation:**
    *   A low CV (near 0) implies a regular lattice.
    *   A high CV implies randomness (Poisson distribution).
*   **Extension:** Compare the ratio of the 1st nearest neighbor to the 2nd nearest neighbor. In a square grid, the 1st neighbor is at distance $d$ and the 2nd (diagonal) is at $d\sqrt{2} \approx 1.41d$. In a hexagonal grid, the ratio is 1 (6 neighbors at distance $d$).

#### C. Neighbor Angle Distribution
*   **The Statistic:** Perform a Delaunay Triangulation or simply find the 4 nearest neighbors. Calculate the angles between vectors pointing to adjacent neighbors.
*   **Interpretation:** Plot a histogram of these angles.
    *   **Square Grid:** Sharp peaks at $90^\circ$ and $180^\circ$.
    *   **Hexagonal Grid:** Peaks at $60^\circ$.
    *   **Random:** Broad, flat distribution.

---

### 2. Spatial Point Pattern Statistics
These are standard statistical methods used in ecology and geography to detect regularity vs. clustering.

#### A. Ripley’s K Function / L-Function
This measures the expected number of points within a distance $r$ of an arbitrary point.
*   **The Statistic:** Calculate $L(r) = \sqrt{K(r)/\pi}$.
*   **Interpretation:**
    *   For a random distribution (CSR), $L(r) - r = 0$.
    *   For a grid, $L(r) - r$ will show distinct **oscillations** (peaks and valleys) corresponding to the lattice spacing shells ($d, d\sqrt{2}, 2d, \dots$).

#### B. Pair Correlation Function (Radial Distribution Function, $g(r)$)
This is the probability of finding a particle at distance $r$, normalized by density.
*   **Interpretation:**
    *   **Square Grid:** Sharp, distinct peaks at $r=1, 1.41, 2, 2.24...$ (relative to lattice constant).
    *   **Distorted Grid:** The peaks broaden but distinct gaps between shells remain visible.
    *   **Random:** Flat line at $g(r)=1$.

---

### 3. Spectral and Global Statistics
These are best for detecting **affine distortions** (shearing, scaling) and checking global periodicity.

#### A. The Structure Factor (2D Fourier Transform)
Treat the points as delta functions in a 2D image and compute the FFT.
*   **Interpretation:**
    *   **Square Grid:** You will see a central peak surrounded by 4 distinct peaks in a square formation (reciprocal lattice).
    *   **Distorted Square:** The peaks will move or smear, but the 4-fold symmetry (or 2-fold if sheared) remains visible in frequency space.
    *   **Statistic:** Calculate the "sharpness" of the peaks or the ratio of energy in the peaks vs. the background noise.

#### B. Principal Component Analysis (PCA) of Local Neighborhoods
*   **Method:** For every point, take the $(x,y)$ coordinates of its neighbors, center them, and run PCA.
*   **Interpretation:** In a square grid, the eigenvalues of the covariance matrix should be roughly equal (isotropic local spread). If the grid is sheared or stretched, the ratio of the first two eigenvalues will deviate from 1.

---

### 4. Topology / Graph Theory Approaches
If the grid is highly warped (e.g., mapped onto a curved surface), distance and angle metrics might fail, but topology remains constant.

#### A. Coordination Number (via Delaunay or Gabriel Graph)
Construct a proximity graph (like a Gabriel Graph or Relative Neighborhood Graph).
*   **The Statistic:** Calculate the mean and mode of the degree (number of connections) of each node.
*   **Interpretation:**
    *   **Square Grid:** The mode degree should be **4** (internal points).
    *   **Hexagonal Grid:** The mode degree is 6.
    *   **Random:** The degree distribution follows a Poisson-like curve.

#### B. Convex Hull Peeling
Iteratively calculate the convex hull of the set, count the points on the hull, remove them, and repeat.
*   **The Statistic:** The number of points on the hull of a grid is roughly $4\sqrt{N_{remaining}}$.
*   **Interpretation:** This tests if the "shape" of the point cloud is consistent with a grid-like packing rather than a spherical clump.

---

### Summary Recommendation

To build a robust classifier, combine these three:

1.  **For Square Symmetry:** Use the **Bond Orientational Order Parameter ($\psi_4$)**. This is the single most indicative statistic for "squareness."
2.  **For Grid Regularity:** Use the **Coefficient of Variation of NN distances**. This detects if it is a grid vs. random noise.
3.  **For Distortion Severity:** Use the **Radial Distribution Function ($g(r)$)**. The width of the first peak tells you how much "jitter" or perturbation exists in the grid positions.