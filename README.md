# MIL 1D-CNN + Multimodal Fusion for Fetal Acidemia Prediction

A Multi-Instance Learning (MIL) 1D Convolutional Neural Network that models the
temporal trajectory of intrapartum cardiotocography (CTG) signals, fused with
clinical metadata and handcrafted signal features in a LightGBM classifier,
to predict **fetal acidemia (umbilical artery pH ≤ 7.15)** during labor.

### 📊 [View the presentation (PDF)](Presentation.pdf) · 📄 [Read the manuscript](Manuscript.docx)

## Results

At a safety-first operating point on the CTU-UHB cohort (546 records, 5-fold out-of-fold predictions):

| Metric | Value |
|---|---|
| Sensitivity (recall) | **0.90** |
| Precision (PPV) | **0.25** |
| Specificity | 0.32 |
| ROC-AUC | 0.74 |
| PR-AUC | 0.48 |

Values are from the saved run in `notebooks/step3_mil_cnn_lightgbm_fusion.ipynb`.

For comparison, the clinical baseline reported in *Cureus* 2025;17(11) gives recall 0.60, precision 0.34.

## Method

```
 CTU-UHB records (FHR + UC, 4 Hz)
        │
        ▼
 STEP 0  signal cleaning ─► handcrafted FHR/UC + clinical features ─────────┐
        │                                                                   │
        ▼                                                                   │
 4-channel full-length signals (FHR, UC, FHR-missing mask, UC-missing mask) │
        │                                                                   │
        ▼                                                                   │
 STEP 2  15-min windows, 5-min stride  ─►  one MIL "bag" per labor          │
        │                                                                   │
        ▼                                                                   │
 STEP 3  1D-CNN ─► 16-d embedding per window                                │
                 ─► aggregated per patient (mean / max / std / latest …)    │
                 ─► 64-d trajectory representation                          │
                                    │                                       │
                                    └──── late fusion (merge on record_id) ◄┘
                                                    │
                                                    ▼
                                    LightGBM (stratified CV) ─► risk score
```

- **Label:** acidemia = umbilical artery pH ≤ 7.15.
- **Leakage control:** post-birth variables (Apgar, base excess, delivery type,
  labor-stage durations, birth weight, …) are excluded from the predictors.
- **Optional pretraining (step 1):** a 1D U-Net trained on expert FHRMA
  annotations detects accelerations/decelerations. In `PRETRAINED_6CH` mode,
  step 2 adds its ACC and DEC outputs as two extra channels (6 instead of 4).

## Repository layout

```
├── Presentation.pdf              project presentation
├── Manuscript.docx               manuscript
├── src/
│   ├── download_data.py          download CTU-UHB from PhysioNet
│   └── step0_build_inputs.py     cleaning + features → input files for steps 2 and 3
├── notebooks/                    run in Google Colab
│   ├── step1_pretrain_acc_dec_detector.ipynb   (optional) ACC/DEC detector pretraining
│   ├── step2_create_mil_bags.ipynb             windowing → MIL bags
│   └── step3_mil_cnn_lightgbm_fusion.ipynb     MIL-CNN + LightGBM fusion and evaluation
├── data/README.md                where to get the data
└── requirements.txt
```

## How to run

### Step 0 — build the inputs (local)

```bash
pip install -r requirements.txt
python src/download_data.py
python src/step0_build_inputs.py
```

This writes:

| File | Used by |
|---|---|
| `ctu_full_length_signals_cache.npz` | step 2 |
| `cnn_tabular_features_final_no_missingness.csv` | step 3 |
| `ctg_features.csv` | step 0 cache (it skips feature extraction on re-runs) |

### Steps 1–3 — Google Colab

The notebooks read and write the Google Drive folder
`MyDrive/Colab Notebooks/Fetal_project/`. Upload the step 0 outputs there, then:

| Step | Notebook | Reads | Writes |
|---|---|---|---|
| 1 *(optional)* | `step1_pretrain_acc_dec_detector` | `CTGDL_FHRMA_ano_csv/*.csv.csv` | `15min_5min_acc_dec_detector.pt` |
| 2 | `step2_create_mil_bags` | `ctu_full_length_signals_cache.npz` (+ `.pt` in 6-channel mode) | `ctu_4channel_mil_bags.pkl` |
| 3 | `step3_mil_cnn_lightgbm_fusion` | `ctu_4channel_mil_bags.pkl`, `cnn_tabular_features_final_no_missingness.csv` | `record_embeddings.pkl`, `cnn_embedding_model.keras`, `hybrid_cnn_lightgbm_patient_features.csv`, predictions |

In step 2, set `EXPERIMENT_MODE = "BASELINE"` (4 channels, the main result)
or `"PRETRAINED_6CH"` (6 channels, needs step 1).

## Data

- [CTU-UHB Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/), PhysioNet (Chudáček et al., 2014)
- FHRMA expert annotations, CSV version from the [CTGDL dataset](https://zenodo.org/records/19510407) (Fridman, 2025, DOI 10.5281/zenodo.19510407). Step 1 only. See [`data/README.md`](data/README.md).

## Limitations

- Small cohort with a strong class imbalance. Many acidemic cases also had
  medical intervention, which shortens or alters their recordings.
- At 25% precision, about 1 in 4 alerts is a true acidemia case. Clinical use
  would need threshold tuning or secondary risk tiers to limit alert fatigue.
- Adding more handcrafted FHR/UC features did not improve results (feature saturation).

## Author

Ronen Levy

## License

Code: [MIT](LICENSE). The datasets keep their own licenses (CTU-UHB: ODC-BY 1.0; CTGDL/FHRMA: CC-BY 4.0 / GPL-3.0).
