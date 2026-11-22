# Data

The data is not stored in this repository.

## 1. CTU-UHB Intrapartum CTG Database (required)

552 intrapartum CTG recordings (FHR + UC at 4 Hz) with umbilical cord blood gas
outcomes, from PhysioNet: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

```bash
python src/download_data.py      # saves to data/CTU_DB/
```

## 2. FHRMA expert annotations (optional, step 1 only)

Step 1 pretrains an acceleration/deceleration detector on expert-annotated
CTG recordings. It expects a folder `CTGDL_FHRMA_ano_csv/` of `*.csv.csv` files
(e.g. `train01.csv.csv`, `test01.csv.csv`), each with the columns:

`fhr, uc, fhr_nan, uc_nan, baseline, acc, dec`

Source: the **CTGDL** multi-source CTG dataset on Zenodo (N. Fridman, 2025),
https://zenodo.org/records/19510407 (DOI: 10.5281/zenodo.19510407).
Download `CTGDL_FHRMA_ano_csv.tar.gz` (48 MB) and extract it into `data/`.
The annotations originally come from the FHRMA dataset (Boudet et al.),
https://github.com/utsb-fmm/FHRMA.

The notebooks expect the extracted folder at
`MyDrive/Colab Notebooks/Fetal_project/CTGDL_FHRMA_ano_csv/` on Google Drive.

Step 1 is only needed for the `PRETRAINED_6CH` experiment in step 2. The main
result uses the 4-channel `BASELINE` mode, which does not need it.
