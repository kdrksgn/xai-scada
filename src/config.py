
import os
import glob

# Paths
BASE_DIR = r"F:\Yeni klasör\ontoXAI"
REPRO_DIR = os.path.join(BASE_DIR, "reproduction")
DATA_DIR_BATADAL = REPRO_DIR  # BATADAL CSVs are here
DATA_DIR_SWAT = os.path.join(BASE_DIR, r"SWaT\SWaT\SWaT.A8_June 2021\For Mark")

# Files
BATADAL_TRAIN = os.path.join(DATA_DIR_BATADAL, "BATADAL_dataset03.csv")
BATADAL_TEST = os.path.join(DATA_DIR_BATADAL, "BATADAL_dataset04.csv")

# SWaT: We will use a glob to find all CSVs
SWAT_FILES = glob.glob(os.path.join(DATA_DIR_SWAT, "**", "*.csv"), recursive=True)

# Training Config
WINDOW_SIZE = 60
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Features (SWaT Standard Tags - Subset based on common usage)
# Note: The CSV header showed FIT101, LIT101, MV101, P101, P102...
# We will auto-detect columns starting with FIT, LIT, MV, P, UV, AIT, DPIT in data_loader
