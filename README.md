# XFAD: Does Explainability Improve False-Alarm Triage in ECG Monitoring?

**Official PyTorch Implementation, Frozen Configuration Module, Pre-Registered Checkpoints & Reproduction Notebooks (`NB00`–`NB07`)**

> **Manuscript:** *Does Explainability Improve False-Alarm Triage in ECG Monitoring? A Pre-Registered, Budget-Matched Evaluation on MIT-BIH and PTB-XL*  
> **Authors:** Eram Mahamud\*, S M Zain (*Faculty of Science and Engineering, Macquarie University, Sydney, NSW, Australia*)  
> **Journal:** *SN Computer Science*

---

## Overview

**XFAD** (*eXplainability-guided False-Alarm Detection*) evaluates whether post-hoc Grad-CAM saliency localisation, signal-quality indices (SQI), and softmax confidence can suppress false arrhythmia alarms raised by a compact, multi-task distilled 1D-CNN student without violating a pre-registered **2% sensitivity-loss budget**.

### Key Empirical Findings

1. **Primary Pre-Registered Test Evaluation (`NB05`):**
   - On validation (`MIT-BIH DS1` validation split: 5 records, 11,616 beats, 815 student alarms), the calibrated two-stage gate met its 2% sensitivity-loss budget (**18.6% false-alarm reduction at 1.78% sensitivity loss**).
   - On the held-out `MIT-BIH DS2` test partition (22 records, 49,687 beats, 2,850 student alarms across 21 records), **no evaluated policy met the pre-registered 2% budget**. The full two-stage gate (`Stage A + Stage B`) achieved **30.7% false-alarm reduction (95% CI 17.0–49.8%) at 13.6% sensitivity loss**, whereas a simple, equally budget-matched **SQI-only baseline** (no learned model, no Grad-CAM, no teacher escalation) achieved **33.2% false-alarm reduction (95% CI 21.6–47.8%) at 12.3% sensitivity loss** (paired bootstrap difference $-2.0\text{ pp}, p = 0.80$).
2. **Post-Hoc Revision Diagnostics (`NB07`):**
   - **Statistical Non-Significance vs. Equivalence (TOST):** Two One-Sided Tests at a post-hoc $\pm 5.0\text{ pp}$ equivalence margin yielded a 90% bootstrap CI of $-20.8\text{ to }+20.2\text{ pp}$ ($p_{\text{TOST}} = 0.451$), demonstrating that $p = 0.80$ reflects inter-record variance rather than formal policy equivalence while confirming no advantage over SQI-only.
   - **Explainability Sample Expansion & Saliency Density ($n = 50 \to n = 250$):** Weight-randomisation Spearman $\rho = 0.056 \pm 0.223$ (95% CI $0.030\text{–}0.084$), paired deletion $\Delta\text{AUC} = +0.184 \pm 0.127$ ($p < 0.001$), and duration-normalised QRS saliency density is **$2.76\times$ ($n=50$) to $2.80\times$ ($n=250$) uniform baseline** ($>3.4\times$ higher than T-wave density $0.80\times\text{–}0.82\times$).
   - **Deconfounded Saliency-Matching Distillation (Protocols A–C):** Isolating the saliency-matched student from its $3.9\times$ alarm-volume surge ($2,850 \to 11,136$ alarms, caused by normal-beat logit-margin compression from $4.72 \to 1.98$) via fixed-beat-pool saliency swap (**3.9% FA reduction**), validation-prevalence matching (**3.9% FA reduction**), and retrospective exact DS2 volume matching ($n=2,850$ alarms, **4.6% FA reduction**) confirms that saliency alignment consistently degrades gate performance.
   - **Leave-One-Record-Out (LORO) & Full-DS1 Calibration (`50,995` DS1 beats):** Conservative 90th-percentile LORO calibration across the 5 validation records halves DS2 sensitivity loss ($13.59\% \to 6.93\%$ at $26.6\%$ FA reduction), and 22-record DS1 LORO yields $16.42\%$ loss ($38.3\%$ FA reduction), confirming that static DS1 thresholds fail the 2% budget due to severe inter-patient covariate shift (KS $D = 0.293, p = 1.3 \times 10^{-33}$ on $1-\text{QRS overlap}$).

---

## Repository Structure

```text
xfad-ecg-triage/
├── src/
│   ├── config.py                  # Single frozen configuration module (splits, seed=42, budgets, hyperparameters)
│   ├── data/                      # MIT-BIH (de Chazal DS1/DS2) & PTB-XL (10-fold) loaders, resampling, NPZ caching
│   ├── models/                    # TeacherA (1D-CNN), TeacherB (1D-ResNet), MultiTaskStudent, KD & Saliency-KD losses
│   ├── explain/                   # 1D Grad-CAM (batched & differentiable), weight-randomisation & deletion AUC checks
│   ├── gate/                      # Two-stage gate (Stage A logistic/fallback scorer, Stage B teacher escalation, calibration)
│   ├── metrics/                   # Patient/record-level bootstrap CIs, DeLong ROC test, McNemar test, Holm-Bonferroni
│   ├── paper/                     # Automated Markdown & LaTeX table generators from result JSONs
│   └── utils/                     # Band-power SQI, Pan-Tompkins QRS detector, class weights, early stopping, IO helpers
├── notebooks/
│   ├── NB00_Setup.ipynb                       # Dataset download, split verification & leakage-safe NPZ cache generation
│   ├── NB01_Teachers.ipynb                    # Train & evaluate specialist Teacher A (MIT-BIH) and Teacher B (PTB-XL)
│   ├── NB02_Student_KD.ipynb                  # Train baseline & saliency-matched multi-task students via KD
│   ├── NB03_Explainability.ipynb              # Compute Grad-CAM sanity checks, deletion faithfulness & QRS localisation
│   ├── NB04_Gate.ipynb                        # Fit Stage-A scorer & calibrate thresholds on DS1 validation under 2% budget
│   ├── NB05_Full_Evaluation.ipynb             # Primary pre-registered held-out evaluation on MIT-BIH DS2 & PTB-XL fold 10
│   ├── NB06_Stretch_Saliency.ipynb            # Saliency-matching KD ablation & Stage-A supplementary statistics
│   └── NB07_1st_Review_Experiments.ipynb      # Post-hoc peer-review experiments (n=250 CIs, 7-subset ablations, TOST, LORO, Protocols A-C)
├── artifacts/
│   ├── checkpoints/               # Frozen pre-registered PyTorch checkpoints (student.pt, student_saliency.pt, teacher_a.pt, teacher_b.pt)
│   ├── results/                   # Primary evaluation JSONs + post-hoc diagnostic JSONs (exp1–exp5)
│   └── tables/                    # Auto-generated Markdown and LaTeX manuscript tables
├── figures/                       # Manuscript Figures 1-4 (PNG and PDF)
├── tests/                         # Pytest suite verifying split disjointness, KD loss, differentiable Grad-CAM, and gate logic
├── requirements.txt
└── pytest.ini
```

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/smzainbd/xfad-ecg-triage.git
cd xfad-ecg-triage
pip install -r requirements.txt
```

### 2. Run the Unit Test Suite

```bash
pytest
```

### 3. Reproducing the Paper & Revision Experiments

- **Fast Verification / Revision Reproduction (`NB07`):**  
  All frozen model checkpoints (`student.pt`, `student_saliency.pt`, `teacher_a.pt`, `teacher_b.pt`) and pre-computed evaluation JSONs are included in `artifacts/`. Opening `notebooks/NB07_1st_Review_Experiments.ipynb` in Google Colab or Jupyter regenerates all revision tables, TOST equivalence tests, LORO calibration transfers, deconfounded saliency protocols, and revised Figures 3 and 4 directly from the frozen checkpoints.
- **Full End-to-End Training (`NB00` $\to$ `NB06`):**  
  Set `os.environ['XFAD_DRY_RUN'] = '0'` in the switch cell at the top of each notebook (`NB00` through `NB06`) to download raw MIT-BIH (`https://physionet.org/content/mitdb/1.0.0/`) and PTB-XL (`https://physionet.org/content/ptb-xl/1.0.3/`) records, build the preprocessed `.npz` caches with patient/record group IDs, and re-run training and evaluation from scratch under `SEED = 42`.

### 4. PhysioNet Open-Access Data Note (No Credentialing Required)

Both datasets evaluated in this study—the **MIT-BIH Arrhythmia Database v1.0.0** (`https://physionet.org/content/mitdb/1.0.0/`) and **PTB-XL v1.0.3** (`https://physionet.org/content/ptb-xl/1.0.3/`)—are **open-access** public PhysioNet repositories distributed under the Creative Commons Attribution 4.0 International License. **No PhysioNet credentialing, CITI training, or Data Use Agreement (DUA) sign-off is required**: running `notebooks/NB00_Setup.ipynb` with `XFAD_DRY_RUN = '0'` downloads the raw records and builds the leakage-safe `.npz` caches immediately out of the box.

---

## Citation

If you find this repository, protocol, or negative-result benchmark useful in your research, please cite our paper:

```bibtex
@article{mahamud2026xfad,
  title={Does Explainability Improve False-Alarm Triage in ECG Monitoring? A Pre-Registered, Budget-Matched Evaluation on MIT-BIH and PTB-XL},
  author={Mahamud, Eram and Zain, S M},
  journal={SN Computer Science},
  year={2026}
}
```
