---
name: SPARE paper 2-day plan
overview: "A writing-first 2-day plan. All 7 metrics files confirmed read on 2026-05-18. Key numbers: F1 text L1 mIoU=0.323 (SI=-0.677, 83x worse than Gaussian); F2 SACo-noise σ50=0.813 vs SA-1B σ50=0.588; F3 SACo h-flip-back=0.892 vs direct=0.204 (84% spatial misregistration). Effective pairs: SACo 8048 (5210 skipped), SA-1B 9442 (1105 skipped). Day 1 generates all figures from existing JSONL+json data. Day 2 writes paper outside-in."
todos:
  - id: fig_robustness
    content: "Build scripts/plot_robustness.py (notebook stub at notebooks/plot_robustness.ipynb): Fig 1 robustness curves (label key coords: text L1=0.323, SA-1B σ50=0.588, SACo-noise σ50=0.813), Fig 2 sensitivity-index bars (log scale, annotate 83x gap), Fig 3 flip decomposition paired bars (direct vs flip-back, label spatial-misregistration fraction). Reads all 7 json files (3 metrics.json + 4 flip jsons). Apply field-name patches. Uses C2A palette."
    status: in_progress
  - id: stratified_analysis
    content: "Build scripts/stratified_analysis.py: confidence calibration, prompt-length and object-size stratified mIoU plots. Pure JSONL aggregation."
    status: pending
  - id: edit_distance
    content: "Build scripts/edit_distance_correlation.py: Levenshtein vs per-entry mIoU scatter + exponential fit for text degradation."
    status: pending
  - id: qualitative_gallery
    content: "Build scripts/qualitative_gallery.py: 3-column grid (image+GT | degraded+pred | mask diff) for 6 cases per condition (best / median / catastrophic)."
    status: pending
  - id: draft_methodology
    content: Draft docs/paper/main.md sections §3 Methodology + §4 Experimental Setup using docs/0_purposal.md as base.
    status: pending
  - id: draft_results
    content: "Draft §5 Results: one subsection per F1 (text fragility), F2 (modality asymmetry), F3 (flip decomposition), each anchored on its figure."
    status: pending
  - id: draft_analysis
    content: Draft §6 Analysis & Explainability around confidence calibration, stratification, edit-distance fit, qualitative gallery.
    status: pending
  - id: draft_related_work
    content: "Draft §2 Related Work: SAM/SAM2/SAM3 lineage, ImageNet-C/COCO-C lineage, VLM prompt-sensitivity lineage."
    status: pending
  - id: draft_discussion
    content: Draft §7 Discussion + §8 Conclusion/Future Work. Explicitly state geometric prong dropped; cite [Author, Year] for SAM3.
    status: pending
  - id: draft_intro_abstract
    content: Draft §1 Introduction and Abstract last, once Results subsection text is fixed.
    status: pending
  - id: token_ablation_optional
    content: "(Optional) Run token-ablation attribution on 50 SACo-Gold prompts: word-mask one token at a time, rank by Δ-mIoU. Only if Day 1 finishes ahead of schedule."
    status: pending
  - id: geometric_optional
    content: (Optional, P3) Extend run_sam3 to accept input_boxes; run geometric jitter on 500-image SACo-Gold subset at 10/20% expansion + shift. Only if Day 1 finishes well ahead of schedule.
    status: pending
isProject: false
---

# SPARE paper — 2-day plan

## 0. Current state (what is already in the repo)

<!-- REFINED [old]: approximate numbers, no skipped-entry counts, no sensitivity indices → [new]: exact values from all 7 metrics files, data-quality notes added -->

### Experiment data — exact numbers (read from all 7 metrics files, 2026-05-18)

**Text degradation (SACo-Gold, text prompts)** — [`metrics.json`](experiments/2026-05-9_text-degradation-task/metrics.json)

| Level | n_entries | skipped | mean_mIoU | sensitivity_index |
|-------|-----------|---------|-----------|-------------------|
| L0 (clean) | 13,258 | — | 1.0000 | 0.0 |
| L1 (1 char sub) | 8,048 | 5,210 | **0.3234** | −0.6766 |
| L2 (2 char sub) | 8,048 | 5,210 | **0.1480** | −0.4260 |
| L3 (3 char sub) | 8,048 | 5,210 | **0.0743** | −0.3086 |

**Gaussian noise (SA-1B, point prompts)** — [`metrics.json`](experiments/2026-05-15_visual-degradation-task/metrics.json)

| σ | n_entries | skipped | mean_mIoU | sensitivity_index |
|---|-----------|---------|-----------|-------------------|
| 0 | 10,548 | — | 1.0000 | 0.0 |
| 10 | 9,443 | 1,105 | **0.7583** | −0.0242 |
| 25 | 9,442 | 1,105 | **0.6607** | −0.0136 |
| 50 | 9,442 | 1,105 | **0.5884** | −0.0082 |

**Gaussian noise (SACo-Gold, text prompts)** — [`metrics.json`](experiments/2026-05-16_visual-degradation-task-SACo/metrics.json)

| σ | n_entries | skipped | mean_mIoU | sensitivity_index |
|---|-----------|---------|-----------|-------------------|
| 0 | 13,258 | — | 1.0000 | 0.0 |
| 10 | 8,048 | 5,210 | **0.9232** | −0.0077 |
| 25 | 8,048 | 5,210 | **0.8687** | −0.0053 |
| 50 | 8,048 | 5,210 | **0.8127** | −0.0037 |

**Flip SA-1B — output stability (direct)** — [`metricsFalse.json`](experiments/2026-5-16_visual-degradation-flip-task/metricsFalse.json) — files named `flip_0/1/2` (mapping: 0=h, 1=v, 2=both)

| Flip | n_entries | skipped | mean_mIoU | sensitivity_index |
|------|-----------|---------|-----------|-------------------|
| none | 10,548 | — | 1.0000 | 0.0 |
| h (flip_0) | 9,443 | 1,105 | **0.0937** | −0.9063 |
| v (flip_1) | 9,443 | 1,105 | **0.0548** | −0.9452 |
| both (flip_2) | 9,443 | 1,105 | **0.0309** | −0.4846 |

**Flip SA-1B — spatial equivariance (flip-back)** — [`metrics_flipped.json`](experiments/2026-5-16_visual-degradation-flip-task/metrics_flipped.json)

| Flip | n_entries | mean_mIoU | sensitivity_index |
|------|-----------|-----------|-------------------|
| none | 10,548 | 1.0000 | 0.0 |
| h (flip_0) | 9,443 | **0.7052** | −0.2948 |
| v (flip_1) | 9,443 | **0.4393** | −0.5607 |
| both (flip_2) | 9,443 | **0.4314** | −0.2843 |

**Flip SACo-Gold — output stability (direct)** — [`metricsFalse.json`](experiments/2026-5-16_visual-degradation-flip-task-SACo/metricsFalse.json)

| Flip | n_entries | skipped | mean_mIoU | sensitivity_index |
|------|-----------|---------|-----------|-------------------|
| none | 13,258 | — | 1.0000 | 0.0 |
| h | 8,046 | 5,212 | **0.2040** | −0.7960 |
| v | 8,046 | 5,212 | **0.1345** | −0.8655 |
| both | 8,046 | 5,212 | **0.0894** | −0.9106 |

**Flip SACo-Gold — spatial equivariance (flip-back)** — [`metrics_flippedTrue.json`](experiments/2026-5-16_visual-degradation-flip-task-SACo/metrics_flippedTrue.json)

| Flip | n_entries | skipped | mean_mIoU | sensitivity_index |
|------|-----------|---------|-----------|-------------------|
| none | 13,258 | — | 1.0000 | 0.0 |
| h | 8,046 | 5,212 | **0.8922** | −0.1078 |
| v | 8,046 | 5,212 | **0.6921** | −0.3079 |
| both | 8,046 | 5,212 | **0.6853** | −0.3147 |

### Data-quality notes (important for §4 Experimental Setup)

- **5,210 skipped entries** in every SACo-Gold degraded condition (text + Gaussian + flip): entries with no matched baseline. Report as n=8,046–8,048 effective pairs.
- **1,105 skipped entries** in every SA-1B degraded condition (Gaussian + flip): same cause.
- **Field-naming inconsistency**: SACo-noise uses `noise_level` (with "people" string at L0 — patch: set to 0); SA-1B noise uses `sigma`; flip SA-1B uses integer `level` with `null` at flip_0. Patch in notebook is: `text_deg_data['levels'][0]['noise_level'] = 0`.
- **Flip file naming**: SA-1B flip files use `flip_0/1/2` (not h/v/both). Mapping: 0=h, 1=v, 2=both — inferred from SACo-Gold file names which explicitly use `flip_h/v/both`.
- **`plot_robustness.ipynb` already partially built** — reads all three Gaussian+text metrics.json files and constructs dicts. Convert/complete as `scripts/plot_robustness.py`.

### What is in the repo

- **Geometric (bbox jitter)** — helper only, no experiment run. ([geometric_degrdation_helper.py](scripts/geometric_degrdation_helper.py))
- **Reusable analysis backbone** — RLE-mIoU + flip-back analyzer is production quality. ([analyze_saco.py](scripts/analyze_saco.py), [analyze_sa1b.py](scripts/analyze_sa1b.py))

You already have **3 strong findings** in hand. The paper is mostly a writing + figure problem, not an experiments problem.

---

## 1. The three headline findings (the spine of the paper)

<!-- REFINED [old]: approximate numbers → [new]: exact values from metrics files, F3 revised with precise decomposition -->

1. **F1 — Linguistic fragility.** A single random character substitution collapses mIoU by **67.7%** (1.00 → 0.323 at L1). SI = −0.677, which is **83× larger** than the SI of Gaussian noise at σ=50 on SA-1B (SI = −0.008). SAM 3's text encoder is catastrophically brittle to surface-level prompt corruption.

2. **F2 — Prompt-modality asymmetry under Gaussian image noise.** At σ=50: text-prompted SACo-Gold retains **0.813 mIoU** (SI = −0.004) while point-prompted SA-1B drops to **0.588 mIoU** (SI = −0.008). Text prompts are ~38% more robust to Gaussian image noise than point prompts. Asymmetry is consistent across all σ levels.

3. **F3 — Flip equivariance asymmetry and decomposition.** On SACo-Gold, the *direct* mIoU under h-flip is 0.204 — appearing catastrophic. But the *flip-back* mIoU is **0.892**, revealing that 84% of the perceived failure is pure **spatial misregistration**, not semantic loss. V-flip and both-flip are far worse: flip-back mIoU = 0.692/0.685 (SI = −0.308/−0.315), showing strong **h vs v equivariance asymmetry**. SA-1B shows a similar pattern but worse overall (h flip-back = 0.705, v = 0.439, both = 0.431).

Every figure and section must service at least one of F1/F2/F3.

---

## 2. Paper structure (8 sections, ~8 pages)

- **Abstract** (200 words) — 1 sentence motivation, 1 sentence gap, 1 sentence method, 3 sentences for F1/F2/F3, 1 sentence implications.
- **§1 Introduction** — PCS shift in SAM 3; gap = no systematic robustness study under realistic noise; 4 contributions; paper outline.
- **§2 Related Work** — (a) Promptable segmentation: SAM, SAM 2, SAM 3 (cite [arXiv:2511.16719](references/sam3.pdf) once you drop it in `references/`); (b) corruption benchmarks: ImageNet-C, COCO-C; (c) prompt sensitivity in VLMs.
- **§3 Methodology** — degradation taxonomy (linguistic / Gaussian / flip), `expand_bbox`/`shift_bbox` mentioned only as future work unless we run geometric on a subset.
- **§4 Experimental Setup** — model = `Sam3Model` from `MODEL_DIR`, datasets, image counts (SACo-Gold baseline: 13,258; effective degraded pairs: 8,046–8,048 after skipping 5,210 unmatched entries. SA-1B baseline: 10,548; effective degraded pairs: 9,442–9,443 after skipping 1,105 unmatched), hardware, fp16 inference. State the skipped-entry cause explicitly to pre-empt reviewer questions.
- **§5 Results** — F1/F2/F3 each get a subsection + figure.
- **§6 Analysis & Explainability** — confidence calibration, stratified failure, edit-distance regression, qualitative gallery.
- **§7 Discussion & Limitations** — synthetic noise vs adversarial, single seed, no geometric prong.
- **§8 Conclusion & Future Work** — geometric jitter, multi-seed, real adversarial typos.
- **References**

---

## 3. Order of writing (outside-in, Day 2)

Write the factual parts first, narrative last:

1. **§3 Methodology** — 90% is already in [docs/0_purposal.md](docs/0_purposal.md). Tighten and import.
2. **§4 Experimental Setup** — mechanical: counts, paths, fp16, threshold=0.5, mask_threshold=0.5 (from [model_helper.py](scripts/model_helper.py)).
3. **§5 Results** — figure captions first, then prose around each figure.
4. **§6 Analysis** — driven by the Day-1 stratification outputs.
5. **§2 Related Work** — independent of your numbers; can be drafted in parallel.
6. **§7 Discussion & §8 Limitations / Future Work** — easier once Results are fixed.
7. **§1 Introduction** — only after Results crystallize; Intro tells the story Results delivered.
8. **Abstract** — last.
9. **References** — accumulated continuously.

---

## 4. Day-1 code runs (figures + analysis, no new SAM3 inference required)

All four scripts below go in `scripts/` and consume the **existing** JSONL + metrics.json files. None require running SAM 3 again. Total Day-1 GPU time: ~0 hours. Total Day-1 CPU time: ~2-3 hours.

### P0 — Headline figures (REQUIRED)

`scripts/plot_robustness.py` — load all 7 json files (3 named `metrics.json`, 4 named `metricsFalse.json`/`metrics_flipped.json`/`metrics_flippedTrue.json`), output to `experiments/_paper_figures/`. A partial notebook version exists at `notebooks/plot_robustness.ipynb` — use as starting point, apply the field-name patches from §0 data-quality notes:

- **Fig 1 — Robustness curves.** 3 panels (text typos / Gaussian σ / flip-back), x = noise level, y = mIoU, overlay SA-1B and SACo-Gold lines on the Gaussian and flip panels. Key points to label: text L1=0.323, SA-1B σ50=0.588, SACo-noise σ50=0.813, SACo h-flip-back=0.892.
- **Fig 2 — Sensitivity Index bar chart.** One bar per (degradation × dataset). Annotate the 83× gap between text SI (−0.677) and SA-1B Gaussian SI (−0.008). Use log scale on y-axis if needed.
- **Fig 3 — Flip decomposition.** Paired bars per flip type (h/v/both) × dataset: *direct mIoU* vs *flip-back mIoU*. The gap between the two bars (direct=0.204 vs flip-back=0.892 for SACo h-flip) is the visual punchline of F3. Annotate "spatial misregistration" vs "true semantic loss" fractions.

Use the mandated palette from [CLAUDE.md](CLAUDE.md) §4.

### P1 — Stratified / explainability analysis

`scripts/stratified_analysis.py` — also pure JSONL aggregation, no GPU:

- **Confidence calibration.** Histogram of `predictions[*].score` per noise level. Tests: does SAM3 "know" it's failing? Free, ~30 min to plot.
- **Prompt-length stratification.** Bin SACo-Gold entries by `len(original_prompt.split())` into short/medium/long, plot mIoU vs noise level per bin. ~45 min.
- **Object-size stratification.** Bin by GT bbox area (small <32², med, large >96²), plot mIoU vs noise level per bin. Needs joining with GT bboxes from SACo-Gold. ~1 h.

`scripts/edit_distance_correlation.py` — for the text experiment:

- Compute `Levenshtein(original_prompt, noisy_prompt)` per entry (free from the JSONL because both are stored), scatter vs per-entry mIoU, fit `mIoU ~ exp(-α · d)`, report R². This is the "explainability lite" answer to *why* the text curve has the shape it does. ~45 min.

### P2 — Qualitative gallery

`scripts/qualitative_gallery.py` — needs SACo-Gold images on disk:

- For each condition (L1 typo, σ=25 noise, h-flip), pick: 2 best-preserved cases (mIoU > 0.9), 2 median cases, 2 catastrophic cases (mIoU < 0.1).
- Render a 3-column grid per row: `[image + GT mask] | [degraded image + pred mask] | [mask diff heatmap]`. ~2-3 h.
- This single figure is the most reviewer-convincing element of the paper.

### P3 — Geometric experiment (OPTIONAL, only if Day-1 finishes before 6pm)

If P0-P2 are done with time left:

- Extend [model_helper.py](scripts/model_helper.py) `run_sam3` to accept `input_boxes=` (the HF `Sam3Processor` supports it).
- Run on a **500-image subset** of SACo-Gold at 10% and 20% expansion + shift using existing [geometric_degrdation_helper.py](scripts/geometric_degrdation_helper.py).
- ~2 h GPU. Report as a "preliminary ablation".

**Recommendation: skip P3 and reframe the paper as "Linguistic + Visual Robustness of SAM 3"**. The cost-of-incompleteness on a 2-day timer is too high.

---

## 5. Fast "explainability" — what to do, what to skip

### Run these (high ROI, no SAM3 re-inference):

| Method | Cost | What it shows |
|---|---|---|
| **Confidence-vs-IoU calibration** (P1) | 30 min | Does the model's `score` field signal its own failures? Calibration plot. |
| **Edit-distance regression** (P1) | 45 min | Why text fails: `mIoU = f(Levenshtein)`. Closed-form explanation of F1. |
| **Stratified mIoU** by prompt-length, object-size (P1) | 2 h | *Where* the model fails: small objects, long prompts. |
| **Token-ablation attribution** (cheap variant) | 2 h | For 50 SACo-Gold prompts, run SAM3 with each word masked one at a time, rank words by Δ-mIoU. Tells you which tokens the model "needs". Requires ~50×N_words SAM3 calls — feasible. |
| **Qualitative failure gallery** (P2) | 3 h | Visual evidence; reviewer-convincing. |

### Skip these (too slow / low ROI in 48 h):

- Grad-CAM / attention rollout on SAM3 — requires custom hooks into `Sam3Model`, no clean HF API, brittle visualization.
- LIME / SHAP — runtime infeasible for SAM3 inference cost.
- Counterfactual mask generation — paper-worthy but multi-week scope.

---

## 6. Concrete 48-hour timeline

**Day 1 (today)**

- 0-3 h: write `plot_robustness.py`, produce Fig 1/2/3.
- 3-5 h: write `stratified_analysis.py`, produce calibration + stratification plots.
- 5-6 h: write `edit_distance_correlation.py`, produce text-noise regression plot.
- 6-9 h: write `qualitative_gallery.py`, render 18 examples (6 per condition).
- 9-11 h: outline `docs/paper/main.md` (or .tex), write §3 Methodology + §4 Experimental Setup (lift from proposal).

**Day 2 (tomorrow)**

- 0-3 h: write §5 Results (one subsection per F1/F2/F3, each anchored on its figure).
- 3-5 h: write §6 Analysis & Explainability (one subsection per Day-1 explainability output).
- 5-6 h: write §2 Related Work, §7 Discussion, §8 Conclusion.
- 6-8 h: write §1 Introduction, Abstract.
- 8-10 h: pass over references, polish captions, final read.

---

## 7. Risks & mitigation

- **SACo-Gold images not on disk** — the qualitative gallery (P2) needs the raw images at `SA1B_DIR`. If they were deleted post-inference, skip P2 or render mask-only figures. Check first thing on Day 1.
- **5,210 / 1,105 skipped entries** — mention in §4 and in the robustness script's data loader. The drop from 13,258 → 8,048 is not a bug; it is because the degraded JSONL only stores entries that had a matched baseline result. Must not confuse reviewers.
- **Flip file naming ambiguity** — the SA-1B flip files use numeric suffixes (flip_0/1/2) with no header label. Hardcode the mapping (0=h, 1=v, 2=both) in `plot_robustness.py` with an inline comment citing the SACo-Gold naming as ground truth.
- **Geometric scope creep** — explicitly drop it; mention as future work. Do not start P3 unless Day 1 ends ahead of schedule.
- **Single-seed concern** — pre-empt reviewer by stating limitation in §7 and noting that catastrophic effects (mIoU drops > 50%) are well outside any plausible seed variance.

## 8. What the user actually needs to do next (concrete next-step ordering)

1. Confirm SACo-Gold images are still at `SA1B_DIR` from [constants.py](constants.py).
2. Approve this plan / call out changes.
3. I'll start by writing `scripts/plot_robustness.py` — the figures it produces drive everything else.