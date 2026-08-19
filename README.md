# HEMC — Hierarchical Eye Movement Classification

Code and notebook supporting the *HEMC* article: a hierarchical two-stage
pipeline for classifying raw gaze samples into eye-movement events.

- **Stage 1** classifies the four macro classes — Fixation (F), Saccade (S),
  Smooth Pursuit (P), Blink (B) — with an ensemble cascade forest
  (**EMCCF++**: multi-scale velocity, acceleration, direction/directional
  change, and dispersion features) followed by a **CRF-Viterbi** sequential
  decoder that removes temporal over-segmentation.
- **Stage 2** takes the fixation samples from Stage 1 and further splits them
  into **microsaccade** vs. **drift/fixation**, using a **ResNet1D**
  classifier trained with a **Focal Loss** and **progressive hard-negative
  mining** .

This repository contains code and the analysis notebook.

## Repository layout

```
HEMCF_Notebook.ipynb   analysis notebook (HMR + RTC, both datasets in one run)
src/hemc/
  data.py         dataset loaders, group-aware / LOSO splitting
  features.py     point-wise kinematics, multi-scale aggregation, EMCCF / EMCCF++ feature sets
  models.py       CascadeForestClassifier, CRF-Viterbi decoder, ResNet-TS
  stage2.py       microsaccade/drift windowing, hard-negative mining
  eval.py         point-wise and event-wise (IoU / oversegmentation) metrics
  utils.py        feature caching, RNG seeding
tests/            pytest unit tests (one file per module, synthetic inputs with known answers)
```

## Setup

Requires Python 3.10+. Stage 2 benefits from an NVIDIA GPU (CPU works, just slower).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# If you have an NVIDIA GPU, install a CUDA-enabled PyTorch build first
# (adjust the cu1xx tag to your driver):
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e .
pytest tests/ -v                 # should pass with no dataset needed
```

## Data setup


```
Data/
  data_hmr/user_<N>/eye_{0,1}.csv     # HMR (public reproduction dataset), 200 Hz
  data_gazecom/subject/participant.csv     # Gazecom (public reproduction dataset), 250 Hz
  data_RTCS2/*.csv                    # RTC: private surgical-training recordings, 200 Hz
```

- **HMR** format (`X_coord`, `Y_coord`,
  `Confidence`, `Pattern`).
- **Gazecom** format (`X_coord`, `Y_coord`,
  `Confidence`, `Pattern`).
- **RTC** format (`gaze x [px]`, `gaze y [px]`, `Class`).

## Running the notebook

Open `HEMCF_Notebook.ipynb` from the **repository root**
(so relative paths to `src/` and `Data/` resolve correctly). The notebook has
one `QUICK` toggle (top cell): `True` for a fast structural sanity check,
`False` for the full-scale run used to produce the article's numbers.

It runs, per dataset:
1. Cascade forest → CRF-Viterbi
   - **HMR** (quantitative reference): event-wise F1 and the
     over-segmentation ratio, before vs. after CRF-Viterbi and the same for **RTC** (real-world "field" recordings)
     
2. (RTC only) Stage 2: fixation vs. microsaccade, ResNet1D + Focal Loss +
   progressive hard-negative mining, evaluated LOSO by participant

## Testing

```bash
pytest tests/ -v
```

Covers: kinematics on synthetic trajectories with closed-form answers,
multi-scale aggregation shapes/edge-padding, cascade forest fit/predict on
separable synthetic data, Viterbi decoding recovering a smooth path from a
flickering probability sequence, event-wise IoU matching / oversegmentation
ratio on hand-constructed sequences, and microsaccade detectors on synthetic
drift+spike signals. No GPU required.


