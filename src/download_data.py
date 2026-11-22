"""
Download the CTU-UHB Intrapartum Cardiotocography Database from PhysioNet
into data/CTU_DB (about 550 records, open access).

Run from the repository root:
    python src/download_data.py
"""
import wfdb

# Downloads every .dat/.hea file of the database into the target folder
wfdb.dl_database("ctu-uhb-ctgdb", dl_dir="data/CTU_DB")
print("Done: records saved in data/CTU_DB")
