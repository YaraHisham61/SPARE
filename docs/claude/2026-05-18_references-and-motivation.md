<!-- AI-GENERATED
     Model  : Claude Sonnet 4.6
     Date   : 2026-05-18
     Prompt : Search for motivation and references to use for the SPARE paper on SAM 3 prompt robustness. Write all findings inside a .md file in docs with links for each paper.
-->

# SPARE — Motivation & References

## Core Motivation (Introduction Story Arc)

The paper's argument flows through three pillars:

**Pillar 1 — Foundation models are deployed before their failure modes are understood.**
SAM 3 introduces Promptable Concept Segmentation (PCS) with a unified Perception Encoder that natively processes text, image exemplars, and geometric prompts. It doubles the accuracy of prior systems on the SA-Co benchmark. But no study has stress-tested its prompt interface under realistic degradation — meaning downstream users (robotics, medical imaging, content creation) inherit hidden fragility.

**Pillar 2 — Existing robustness work covers SAM 1 only, using geometric prompts only.**
Wang et al. [2023] and the BREPS framework [Moskalenko et al., 2026] study point/box prompt sensitivity on SAM 1/2, but SAM 3's text-prompt channel is entirely unstudied. RobustSAM [Chen et al., 2024] proposes a fix for image corruption sensitivity but never tests text prompt corruption. This leaves a critical gap: nobody has measured cross-modal sensitivity asymmetry (text vs. visual vs. geometric noise on the same model).

**Pillar 3 — The ImageNet-C paradigm needs extension to promptable segmentation.**
Hendrycks & Dietterich [2019] established the corruption robustness benchmark methodology for classifiers. SPARE applies this paradigm to multi-modal prompts, introducing a Sensitivity Index S = Δaccuracy / Δnoise that enables cross-modality comparison.

---

## Three Headline Findings (Paper Spine)

| ID | Finding | Key Numbers |
|----|---------|-------------|
| **F1** | **Linguistic fragility.** A single random character substitution collapses mIoU by 68%. SAM 3's text encoder is brittle. | mIoU: 1.00 → 0.32 → 0.15 → 0.07 at L1/L2/L3 |
| **F2** | **Prompt-modality asymmetry under image noise.** Text prompts are *more* robust to Gaussian image noise than geometric point prompts. | At σ=50: SACo-Gold (text) = 0.81 vs SA-1B (point) = 0.59 |
| **F3** | **Flip equivariance asymmetry.** SAM 3 is nearly h-flip equivariant but not v-flip equivariant. Decomposing direct drop into spatial mis-registration vs true semantic loss is novel. | H-flip equivariance mIoU = 0.89; v/both ≈ 0.69 |

---

## §2a — Promptable Segmentation Lineage

### SAM — Segment Anything
**[Kirillov et al., 2023]** — ICCV 2023
- arXiv: [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
- ICCV PDF: [https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)
- **Why cite:** Original SAM paper. Introduced promptable segmentation with points/boxes/masks, the SA-1B dataset (11M images, 1.1B masks). Our SA-1B data subset comes from this.

### SAM 2 — Segment Anything in Images and Videos
**[Ravi et al., 2024]** — Meta FAIR, 2024
- arXiv: [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)
- **Why cite:** Extended SAM to video with streaming memory. 6x faster than SAM on images. Cite for lineage context between SAM and SAM 3.

### SAM 3 — Segment Anything with Concepts *(model under test)*
**[Peng et al., 2025]** — ICLR 2026
- arXiv: [https://arxiv.org/abs/2511.16719](https://arxiv.org/abs/2511.16719)
- Meta AI page: [https://ai.fb.com/research/publications/sam-3-segment-anything-with-concepts/](https://ai.fb.com/research/publications/sam-3-segment-anything-with-concepts/)
- GitHub: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- OpenReview: [https://openreview.net/forum?id=r35clVtGzw](https://openreview.net/forum?id=r35clVtGzw)
- **Why cite:** The model under test. Introduced PCS, unified Perception Encoder accepting text/image/geometric prompts, SA-Co benchmark (270K unique concepts). Our `Sam3Model` is loaded from this checkpoint.

### SA-Co/Gold Benchmark *(our evaluation dataset)*
**[Meta, 2025]** — HuggingFace Dataset
- Dataset: [https://huggingface.co/datasets/facebook/SACo-Gold](https://huggingface.co/datasets/facebook/SACo-Gold)
- Eval docs: [https://mintlify.com/facebookresearch/sam3/evaluation/saco-gold](https://mintlify.com/facebookresearch/sam3/evaluation/saco-gold)
- **Why cite:** 156K image-NP pairs, triple-annotated across 7 subsets. We use the SA-1B captioner subset (13,258 pairs, 30,306 masks). Official metric = cgF1; we use mIoU for degradation comparison.

---

## §2b — Corruption & Robustness Benchmarks

### ImageNet-C — Benchmarking Neural Network Robustness
**[Hendrycks & Dietterich, 2019]** — ICLR 2019
- arXiv: [https://arxiv.org/abs/1903.12261](https://arxiv.org/abs/1903.12261)
- PDF: [https://web.engr.oregonstate.edu/~tgd/publications/hendrycks-dietterich-benchmarking-neural-network-robustness-to-common-corruptions-and-perturbations-iclr2019.pdf](https://web.engr.oregonstate.edu/~tgd/publications/hendrycks-dietterich-benchmarking-neural-network-robustness-to-common-corruptions-and-perturbations-iclr2019.pdf)
- GitHub: [https://github.com/hendrycks/robustness](https://github.com/hendrycks/robustness)
- **Why cite:** Introduced ImageNet-C/ImageNet-P. 15 corruption types at 5 severity levels. Methodological ancestor of our degradation taxonomy. Justifies the "systematic corruption levels + sensitivity index" approach.

### LAION-C — Out-of-Distribution Benchmark for Web-Scale Vision Models
**[Li et al., 2025]** — ICML 2025
- arXiv: [https://arxiv.org/abs/2506.16950](https://arxiv.org/abs/2506.16950)
- PMLR: [https://proceedings.mlr.press/v267/li25aw.html](https://proceedings.mlr.press/v267/li25aw.html)
- **Why cite:** Shows traditional ImageNet-C corruptions are no longer OOD for large-scale models. Motivates why SAM3-specific corruption testing is necessary rather than relying on generic benchmarks.

---

## §2c — SAM Robustness Studies (Direct Competitors / Precursors)

### Empirical Robustness of SAM
**[Wang, Zhao & Petzold, 2023]** — arXiv 2023
- arXiv: [https://arxiv.org/abs/2305.06422](https://arxiv.org/abs/2305.06422)
- **Gap we fill:** Tested SAM 1 under 15 corruption types (noise, blur, weather, digital). Found SAM robust to most corruptions except blur. **Only geometric prompts (points/boxes); no text prompts; SAM 1 only, not SAM 3.**

### RobustSAM — Segment Anything Robustly on Degraded Images
**[Chen, Vong, Kuo, Ma & Wang, 2024]** — CVPR 2024 (Highlight)
- arXiv: [https://arxiv.org/abs/2406.09627](https://arxiv.org/abs/2406.09627)
- CVPR page: [https://openaccess.thecvf.com/content/CVPR2024/html/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.html)
- Project page: [https://robustsam.github.io/](https://robustsam.github.io/)
- **Gap we fill:** Proposes anti-degradation modules for SAM + 688K Robust-Seg dataset. **Proposes a fix but never studies how prompt modality mediates degradation sensitivity.** We show text prompts are inherently more robust to image noise than point prompts.

### BREPS — Bounding-Box Robustness Evaluation of Promptable Segmentation
**[Moskalenko et al., 2026]** — AAAI 2026
- arXiv: [https://arxiv.org/abs/2601.15123](https://arxiv.org/abs/2601.15123)
- AAAI proceedings: [https://ojs.aaai.org/index.php/AAAI/article/view/37757](https://ojs.aaai.org/index.php/AAAI/article/view/37757)
- GitHub: [https://github.com/emb-ai/BREPS](https://github.com/emb-ai/BREPS)
- **Gap we fill:** Tests bbox prompt sensitivity via adversarial bbox generation on 10 datasets; found high inter-user variability. **Tests within geometric prompts only; no cross-modality comparison; SAM 1/2 only, not SAM 3.**

---

## §2d — Text/Prompt Sensitivity in VLMs

### PARC — Quantitative Framework for VLM Symmetries
**[Schmalfuss et al., 2025]** — CVPR 2025
- arXiv: [https://arxiv.org/abs/2506.14808](https://arxiv.org/abs/2506.14808)
- CVPR PDF: [https://openaccess.thecvf.com/content/CVPR2025/papers/Schmalfuss_PARC_A_Quantitative_Framework_Uncovering_the_Symmetries_within_Vision_Language_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Schmalfuss_PARC_A_Quantitative_Framework_Uncovering_the_Symmetries_within_Vision_Language_CVPR_2025_paper.pdf)
- **Why cite:** Framework measuring VLM sensitivity to systematic prompt variations in both language and vision domains. Found VLMs mirror LLM text sensitivity in the vision domain. Supports our cross-modal framing.

### SCAM — Real-World Typographic Robustness Evaluation
**[Materzynska et al., 2025]** — arXiv 2025
- arXiv: [https://arxiv.org/abs/2504.04893](https://arxiv.org/abs/2504.04893)
- **Why cite:** 1,162 real-world typographic attack images; found up to 42% performance degradation from text *in* images. **Contrast:** we study text *prompts* (user input), not text *in* images (adversarial visual content). Cite to distinguish our attack surface.

### Defense-Prefix for Typographic Attacks on CLIP
**[Luo et al., 2023]** — arXiv 2023
- arXiv: [https://arxiv.org/abs/2304.04512](https://arxiv.org/abs/2304.04512)
- **Why cite:** Proposes prefix-token defense for CLIP text encoder against typos. Relevant because SAM 3's text encoder likely inherits CLIP-like sensitivity to character-level noise.

### MVP — Modeling Variants of Prompts for VLMs
**[2025]** — arXiv 2025
- arXiv: [https://arxiv.org/abs/2503.08229](https://arxiv.org/abs/2503.08229)
- **Why cite:** Models distributions of diverse prompt structures using VAEs for enhanced robustness. Aligns with Finding F1: prompt wording matters enormously.

### TuneVLSeg — Prompt Tuning Benchmark for VL Segmentation
**[2024]** — arXiv 2024
- arXiv: [https://arxiv.org/abs/2410.05239](https://arxiv.org/abs/2410.05239)
- **Why cite:** Found textual prompt tuning struggles under domain shifts while visual prompt tuning is more robust. Aligns with Finding F2: text prompts fragile at character level, but more stable under image noise.

---

## §2e — Spatial Equivariance (for Finding F3)

### PreCM — Rotation Equivariant Convolution for Semantic Segmentation
**[Liu et al., 2024]** — arXiv 2024
- arXiv: [https://arxiv.org/abs/2411.01624](https://arxiv.org/abs/2411.01624)
- **Why cite:** Shows 4.5–10.6% IoU improvement from enforcing rotation equivariance in segmentation. We test whether SAM 3 already exhibits flip equivariance (partially: horizontal yes, vertical no). Frames F3 against what a well-designed equivariant model would do.

---

## §6 / §7 — Adversarial Context (Discussion Positioning)

### Black-box Adversarial Attack on SAM
**[Zhang et al., 2023]** — arXiv 2023
- arXiv: [https://arxiv.org/abs/2310.10010](https://arxiv.org/abs/2310.10010)
- **Why cite:** Demonstrated adversarial attacks on SAM image encoder. Cite in Discussion to contrast our non-adversarial (natural corruption) approach and bound the security implications.

### Practical Region-level Attack against SAM
**[Zheng et al., 2024]** — arXiv 2024
- arXiv: [https://arxiv.org/abs/2404.08255](https://arxiv.org/abs/2404.08255)
- **Why cite:** Region-level attacks work regardless of prompt placement. Cite to argue our findings may underestimate real-world risk since we only test natural noise, not adversarial examples.

---

## Positioning Summary: Why SPARE is Novel

Three gaps no prior work fills simultaneously:

| Gap | Prior Work | SPARE |
|-----|-----------|-------|
| SAM 3 tested | Wang [2023], Chen [2024], Moskalenko [2026] all test SAM 1/2 | ✓ First robustness study of SAM 3 |
| Cross-modal sensitivity | BREPS = geometric only; PARC = VLM text only | ✓ Text vs. geometric vs. visual noise on same model |
| Flip equivariance decomposition | No prior work disentangles spatial misalignment from semantic failure | ✓ Direct mIoU vs. flip-back mIoU decomposition |

---

## Full BibTeX Entries

```bibtex
@inproceedings{kirillov2023sam,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  booktitle = {ICCV},
  year      = {2023},
  url       = {https://arxiv.org/abs/2304.02643}
}

@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024},
  url     = {https://arxiv.org/abs/2408.00714}
}

@inproceedings{peng2025sam3,
  title     = {SAM 3: Segment Anything with Concepts},
  author    = {Peng, Fanbo and others},
  booktitle = {ICLR},
  year      = {2026},
  url       = {https://arxiv.org/abs/2511.16719}
}

@inproceedings{hendrycks2019benchmarking,
  title     = {Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author    = {Hendrycks, Dan and Dietterich, Thomas},
  booktitle = {ICLR},
  year      = {2019},
  url       = {https://arxiv.org/abs/1903.12261}
}

@article{wang2023empirical,
  title   = {An Empirical Study on the Robustness of the Segment Anything Model ({SAM})},
  author  = {Wang, Yuqing and Zhao, Yun and Petzold, Linda},
  journal = {arXiv preprint arXiv:2305.06422},
  year    = {2023},
  url     = {https://arxiv.org/abs/2305.06422}
}

@inproceedings{chen2024robustsam,
  title     = {{RobustSAM}: Segment Anything Robustly on Degraded Images},
  author    = {Chen, Wei-Ting and Vong, Yu-Jiet and Kuo, Sy-Yen and Ma, Sizhuo and Wang, Jian},
  booktitle = {CVPR},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.09627}
}

@inproceedings{moskalenko2026breps,
  title     = {{BREPS}: Bounding-Box Robustness Evaluation of Promptable Segmentation},
  author    = {Moskalenko, Anton and others},
  booktitle = {AAAI},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.15123}
}

@inproceedings{schmalfuss2025parc,
  title     = {{PARC}: A Quantitative Framework Uncovering the Symmetries within Vision Language Models},
  author    = {Schmalfuss, Jenny and others},
  booktitle = {CVPR},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.14808}
}

@article{materzynska2025scam,
  title   = {{SCAM}: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models},
  author  = {Materzynska, Joanna and others},
  journal = {arXiv preprint arXiv:2504.04893},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.04893}
}

@inproceedings{li2025laionc,
  title     = {{LAION-C}: An Out-of-Distribution Benchmark for Web-Scale Vision Models},
  author    = {Li, Jiawei and others},
  booktitle = {ICML},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.16950}
}
```
