# SAM3 Robustness Evaluation - Degradation Results Report
Generated: 2026-05-12 12:35:10
Total evaluations: 320
Unique samples: 5

## OVERALL PERFORMANCE

Baseline mIoU (no degradation): 0.8462 +/- 0.0355
Worst-case mIoU: 0.0000 (config: geo_30_15_ling_20_vis_50)
Average mIoU drop: 0.3376
Maximum mIoU drop: 0.9108
Performance retention: 60.10% on average

## GEOMETRIC DEGRADATION IMPACT

Geometric degradation applies bounding box perturbations (expansion & center shift).

        degraded_miou                         miou_drop sensitivity
                 mean     std     min     max      mean        mean
geo_exp                                                            
0              0.7024  0.1043  0.4100  0.9060    0.1438      0.0028
10             0.5687  0.1416  0.2495  0.7900    0.2775      0.0025
20             0.4483  0.1509  0.0416  0.7139    0.3979      0.0025
30             0.3167  0.1580  0.0000  0.6337    0.5312      0.0025

Average mIoU drop by expansion level:
  - Expansion 0% (baseline): 0.1438 drop
  - Expansion 10%: 0.2775 drop
  - Expansion 20%: 0.3979 drop
  - Expansion 30%: 0.5312 drop

## LINGUISTIC DEGRADATION IMPACT

Linguistic degradation applies keyboard-adjacent typos to text prompts.

         degraded_miou                         miou_drop sensitivity
                  mean     std     min     max      mean        mean
ling_err                                                            
0.0             0.5527  0.1955  0.0241  0.9014    0.2936      0.0028
0.1             0.5249  0.2048  0.0019  0.9060    0.3213      0.0024
0.2             0.4773  0.2040  0.0000  0.8186    0.3700      0.0027
0.3             0.4812  0.1886  0.0000  0.8269    0.3657      0.0025

Average mIoU drop by error rate:
  - Error rate 0.0% (baseline): 0.2936 drop
  - Error rate 10.0%: 0.3213 drop
  - Error rate 20.0%: 0.3700 drop
  - Error rate 30.0%: 0.3657 drop

## VISUAL DEGRADATION IMPACT

Visual degradation applies Gaussian noise (AWGN) to images at various sigma levels.

          degraded_miou                         miou_drop sensitivity
                   mean     std     min     max      mean        mean
vis_sigma                                                            
0                0.6240  0.1590  0.2716  0.9060    0.2222      0.0027
10               0.5816  0.1571  0.1826  0.8441    0.2646      0.0024
25               0.4710  0.1783  0.0383  0.7273    0.3753      0.0027
50               0.3595  0.1917  0.0000  0.7016    0.4884      0.0025

Average mIoU drop by noise level (sigma):
  - Noise sigma=0 (baseline): 0.2222 drop
  - Noise sigma=10: 0.2646 drop
  - Noise sigma=25: 0.3753 drop
  - Noise sigma=50: 0.4884 drop

## CROSS-MODAL SENSITIVITY ANALYSIS

Analysis of combined degradation effects across modalities.

Most robust configuration:
  - mIoU drop: 0.0000
  - Config: geo_0_0_ling_10_vis_0
  - Degradation: geo_exp=0, ling_err=0.1, vis_sigma=0

Least robust configuration:
  - mIoU drop: 0.9108
  - Config: geo_30_15_ling_20_vis_50
  - Degradation: geo_exp=30, ling_err=0.2, vis_sigma=50

## SENSITIVITY RANKING

Average sensitivity to each degradation type (higher = less robust):

1. Visual: 0.002561
2. Linguistic: 0.002533
3. Geometric: 0.002532

## TOP 5 MOST SENSITIVE CONFIGURATIONS

                     config  sensitivity  miou_drop
128    geo_0_0_ling_0_vis_0     0.013639     0.0136
0      geo_0_0_ling_0_vis_0     0.012954     0.0130
197  geo_0_0_ling_10_vis_10     0.004657     0.1397
257   geo_0_0_ling_0_vis_10     0.004656     0.0931
64     geo_0_0_ling_0_vis_0     0.004526     0.0045

## RECOMMENDATIONS FOR IMPROVING ROBUSTNESS

Based on the sensitivity analysis:

PRIORITY 1: Visual Robustness
  - Issue: Model is highly sensitive to image noise
  - Solutions:
    * Train with heavy image augmentations (AWGN, blur, compression)
    * Implement denoising preprocessing (e.g., bilateral filter)
    * Use domain randomization for visual degradation
    * Consider noise-adaptive normalization layers

Cross-Modal Strategy:
  - Combine robustness improvements across all three modalities
  - Validate on combinations of degradations, not just individual types
  - Consider ensemble methods that vote across modalities

Testing and Validation:
  - Test with real-world degradation patterns (not just synthetic)
  - Evaluate on in-the-wild data with natural variations
  - Monitor degradation performance in production deployments
