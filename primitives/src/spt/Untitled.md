
| Effect                   | Routine                 | Controls       | Disable |
| ------------------------ | ----------------------- | -------------- | ------- |
| Lens distortion (radial) | apply_radial_distortion | k1=[-0.2, 0.2] |         |
1. **Lens distortion (radial)**
   **Entry**; radial_distortion
   **Controls**:
    - k1 = [-0.2, 0.2]
    - k2 = [-0.02, 0.02]
   **Disable**:
    - k1=0 and k2=0
2. **Rolling shutter skew**
   **Entry**; rolling_shutter_skew
   **Controls**:
    - strength = [0, 0.05]
   **Disable**:
    - strength=0
3. **CFA + Demosaicing**
   **Entry**; cfa_and_demosaic
   **Controls**:
    - None
   **Disable**:
    - TODO: Add flag at composition point
 
