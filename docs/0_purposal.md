# Analyze Prompt Sensitivity for SAM3
## Problem definition:
Segment Anything Model 3 (SAM 3) has shifted the paradigm of computer vision from geometric edge-detection to Promptable Concept Segmentation (PCS). By utilizing a unified Perception Encoder, SAM 3 natively processes text, image examples, and geometric prompts (points/boxes) simultaneously.

However, it is currently unknown how SAM 3’s unified architecture behaves under systematic prompt degradation. 

## Aim:
* To stress-test SAM 3 against simulated human error, determining the exact thresholds where linguistic, geometric, and cross-modal noise causes segmentation failure.
* Quantify accuracy drops caused by text-based noise (typos, complex attributes).
* Measure resilience to spatial noise e.g., loose bounding boxes…etc. 

## Systematic Methodology (Evaluation Plan) 
We will use a SA-1B dataset, and we will induce different types of degradation :
* **Geometric Degradation**:
    * _Method:_ We will programmatically jitter the "perfect" ground-truth bounding boxes.
    * _Levels:_ We will test 10% and 20% expansion/ pixel shift from the object center.
* **Linguistic Degradation (Textual Noise)**:
    * _Method_: We will use a script to inject character-level errors into the prompt text.
    * _Levels_: 1, 2, and 3 random character substitutions (typos) per object label.
**Visual Degradation**:
    * _Method_: We will inject additive White Gaussian Noise (WGN) into the source images to simulate sensor interference or low-light conditions.
    * _Levels_: We will test three standard deviation ($\sigma$) intensities: $\sigma$ = 10 , $\sigma$ = 25, and $\sigma$ = 50.
## Metrics:
1. **Mean Intersection over Union (mIoU)**: To measure spatial overlap between the SAM 3 generated mask and the original Ground Truth mask.
* **Comparative Sensitivity Analysis**: We will calculate a Sensitivity Index for each noise type by $$S = \frac{\triangledown\text{ accuracy}}{\triangledown\text{ noise level}}$$ 
    
    This will allow us to state something like: “SAM 3 is 2x more sensitive to spatial jitter than it is to spelling errors.”

## References:

[SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)