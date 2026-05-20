# SAM3 Robustness Evaluation - Extended Dataset Analysis
Generated: 2026-05-12 12:40:49
Samples evaluated: 20
Total configurations: 1280

## DATA COVERAGE

Unique samples: 20
Configurations per sample: 64
Total evaluation runs: 1280

## OVERALL PERFORMANCE

Baseline mIoU (no degradation):
  Mean: 0.8359
  Std:  0.0545
  Range: [0.7522, 0.9300]

Worst-case mIoU: 0.0000
  Config: geo_30_15_ling_20_vis_50
  Sample: synthetic_000

Average mIoU drop: 0.3332
Maximum mIoU drop: 0.9772
Performance retention: 60.14%

## DEGRADATION SENSITIVITY ANALYSIS

Geometric Degradation:
  Expansion 0%: 0.1449 avg drop
  Expansion 10%: 0.2730 avg drop
  Expansion 20%: 0.3947 avg drop
  Expansion 30%: 0.5201 avg drop
  Sensitivity: 0.002511

Linguistic Degradation:
  Error rate 0.0%: 0.2960 avg drop
  Error rate 10.0%: 0.3167 avg drop
  Error rate 20.0%: 0.3533 avg drop
  Error rate 30.0%: 0.3667 avg drop
  Sensitivity: 0.002505

Visual Degradation:
  Noise sigma=0: 0.2274 avg drop
  Noise sigma=10: 0.2689 avg drop
  Noise sigma=25: 0.3575 avg drop
  Noise sigma=50: 0.4789 avg drop
  Sensitivity: 0.002518

## SENSITIVITY RANKING

1. Visual: 0.002518
2. Geometric: 0.002511
3. Linguistic: 0.002505

## PER-SAMPLE STATISTICS

              degraded_miou                         miou_drop baseline_miou
                       mean     std     min     max      mean          mean
sample                                                                     
synthetic_000        0.4939  0.1994  0.0000  0.8120    0.3332        0.8249
synthetic_001        0.5663  0.1959  0.1734  0.9060    0.3397        0.9060
synthetic_002        0.5173  0.1971  0.0019  0.8409    0.3337        0.8511
synthetic_003        0.4749  0.1904  0.0241  0.8180    0.3441        0.8190
synthetic_004        0.4927  0.2096  0.0319  0.8266    0.3375        0.8302
synthetic_005        0.4803  0.2110  0.0000  0.8089    0.3407        0.8203
synthetic_006        0.4700  0.1817  0.0147  0.7859    0.3159        0.7859
synthetic_007        0.5970  0.1941  0.0840  0.9100    0.3238        0.9209
synthetic_008        0.6178  0.1753  0.1527  0.9151    0.3122        0.9300
synthetic_009        0.4249  0.1985  0.0000  0.7522    0.3281        0.7522
synthetic_010        0.4264  0.1844  0.0000  0.7397    0.3380        0.7606
synthetic_011        0.4504  0.1819  0.0414  0.7768    0.3321        0.7824
synthetic_012        0.4631  0.1741  0.0580  0.7763    0.3132        0.7763
synthetic_013        0.5132  0.1839  0.0282  0.8523    0.3391        0.8523
synthetic_014        0.5094  0.1985  0.1479  0.8588    0.3494        0.8588
synthetic_015        0.5795  0.2034  0.0002  0.9285    0.3490        0.9285
synthetic_016        0.4836  0.1819  0.0000  0.8057    0.3241        0.8075
synthetic_017        0.5002  0.1803  0.0177  0.8358    0.3356        0.8358
synthetic_018        0.5263  0.2073  0.0000  0.8438    0.3487        0.8749
synthetic_019        0.4753  0.1844  0.0313  0.8006    0.3252        0.8006

## CONCLUSION

Evaluation completed with 20 samples.
Dataset provides 1280 data points for robustness analysis.
Most sensitive degradation type: Visual
