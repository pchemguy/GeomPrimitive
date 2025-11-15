
| Effect                   | Routine                 | Controls       | Disable |
| ------------------------ | ----------------------- | -------------- | ------- |
| Lens distortion (radial) | apply_radial_distortion | k1=[-0.2, 0.2] |         |
1. **Lens distortion (radial)**
   **Entry**; radial_distortion
   **Controls**:
    - k1 - Range: [-0.2, 0.2]. Disable - k1=0.
    - k2 - Range: [-0.02, 0.02]. Disable - k2=0
2. **Rolling shutter skew**
   **Entry**; rolling_shutter_skew
   **Controls**:
    - strength - Range: [0, 0.05]. Disable - strength=0.
3. **CFA + Demosaicing**
   **Entry**; cfa_and_demosaic
   **Controls**:
    - None - Range: on/off. Disable: TODO: Add flag at composition point.
4. **Sensor noise: PRNU + FPN + shot/read noise (ISO-dependent)**
   **Entry**; sensor_noise
   **Controls**:
    - General (scales all noise amplitudes): iso_level - Range: [0.5, 2]. Disable all noises - iso_level=0
    - PRNU: profile.base_prnu_strength - Range: [0, 0.01]. Disable - profile.base_prnu_strength=0
    - FPN:
        - profile.base_fpn_row - Range: [0, 0.01]. Disable - profile.base_fpn_row=0
        - profile.base_fpn_col - Range: [0, 0.01]. Disable - profile.base_fpn_col=0
    - short: profile.base_shot_noise - Range: [0, 0.02]. Disable - profile.base_shot_noise=0
    - read: profile.base_read_noise - Range: [0, 0.01]. Disable - profile.base_read_noise=0
5. **ISP denoise + sharpening**
   **Entry**; isp_denoise_and_sharpen
   **Controls**:
    - Sharpen: profile.sharpening_amount - TODO: Range/Disable - needs clarification
    - Additional controls - TODO: needs clarification.
6. **Vignette + Color warmth**
   **Entry**; vignette_and_color
   **Controls**:
    - Vignette: profile.vignette_strength - Range: [0, 0.5]. Disable - profile.vignette_strength=0
    - warmth: profile.color_warmth - Range: [0, 0.2]. Disable - profile.color_warmth=0
7. **JPEG round-trip**
   **Entry**; jpeg_roundtrip
   **Controls**:
    - None - Range: on/off. Disable - apply_jpeg=False (composition point).
 
 
