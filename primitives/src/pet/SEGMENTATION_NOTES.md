Before making histogram, I want to find inner hot spots:

1) Select top 10th percentile based on Lab-L.
2) Cluster with automatically adjusted params.
3) For each cluster, identify bounding box, covering 95% percent of pixels within cluster.
4) Calculate box center, and the radius of circumscribed circle (bounding circle, Rb)
5) Calculate mean saturation of pixels within the cluster (belonging to bottom 10th percentile only); calculate mean saturation for the torus Rb to 2Rb.

- have statistically lower saturation than mean within the region covered by a circle that is three times the size of candidate defective area. The defective area

Next, I want you to find minimal area bbox for the final mask. The box must be aligned with the mask to minimize area, not axes. Draw bbox in addition to final mask.

Within the final area calculate the mask, that intersects bottom 10th percentile S-HSV with union of top 10% and bottom 10% L-Lab


I updated thresholds as follows
s_p25, l_p10, l_p85

I need a detailed README section, describing this pipeline.

Include consideration that SxL mask attempts to isolate glare points, low information points (low S AND L), and, importantly, blood splashes where lower saturation with higher L is expected. Generally, glares + blood splash areas are expected to result in bimodal S and L distribution with the final Mask. Also include crafted todo:

### TODO 

Consider performing statistical analysis, identifying and separating the two distribution and setting thresholds accordingly. The current manual threshold settings in pet_segmentation_composite3.py show close to optimal separation with thresholds on low S and high L roughly matching corresponding separation thresholds. The resulting mask would need morphological enhancement and identification of smaller inner spots (not bordering with mask outer border). The inner clusters may need to be inpainted.

Classification of inner clusters as glares might be something as follows:
- Identify cluster size (95% percentile distance between any two points)
- Identify minimum distance from cluster center to area outer border. It should be at least twice the size of the cluster.
- Identify minimum area bounding box, calculate mu1 and stddev1 (perhaps, both S and L) for masked pixels only. Double bbox sizes and calculate mu2 and stddev2 for non-masked pixels. If abs(mu2-mu1) > 2*(stdev1+stddev2), consider marking for inpainting.
  