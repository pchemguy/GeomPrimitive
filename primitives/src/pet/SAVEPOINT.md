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
4. Grid data analysis
5. Downstream tasks

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
