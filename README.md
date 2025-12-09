# OntoXAI-LSTM Reproduction

This repository contains a reproduction of the **OntoXAI-LSTM** framework for explainable anomaly detection in Industrial Control Systems (ICS).

## Overview
OntoXAI-LSTM combines LSTM-based temporal anomaly detection with a protocol-aware semantic layer. This allows it not only to detect attacks in SCADA networks but also to provide operator-friendly explanations linked to specific control commands (e.g., Modbus Write Coil).

## Features
- **LSTM Modeling**: Sequence-based deep learning for multivariate time-series anomaly detection.
- **SHAP Integration**: Stabilized feature importance to identify which sensors caused an alert.
- **Ontology Mapping**: Semantic layer to map low-level sensor tags (e.g., `FIT101`) to high-level protocol commands.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ontoxai-reproduction.git
   cd ontoxai-reproduction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Data**:
   - Ensure the BATADAL dataset CSVs (`BATADAL_dataset03.csv`, `BATADAL_dataset04.csv`) are in the project root or configured in `src/config.py`.

2. **Run Pipeline**:
   The `src/main.py` script executes the full pipeline: Data Loading -> Training -> Evaluation -> Explanation.
   ```bash
   python src/main.py
   ```

## Project Structure
- `src/`: Source code directory.
  - `main.py`: Main execution script.
  - `model.py`: LSTM model architecture (TensorFlow/Keras).
  - `data_loader.py`: Data preprocessing and windowing logic.
  - `explain.py`: SHAP-based explanation module.
  - `ontology.py`: Semantic mapping logic.
  - `config.py`: Configuration paths and parameters.
- `requirements.txt`: Python package dependencies.

## Results
This implementation aims to reproduce the results reported in the paper:

> "The resulting dataset contains 14 attack sessions, of which 98% involve Modbus FC=05 (Write Coil) on actuators. OntoXAI-LSTM achieves F1 = 0.99, recall = 1.00 on all FC=05 attacks, and correctly maps the top-3 SHAP features to the targeted actuator via ontology reasoning in every case."

Resources including Zeek scripts, alignment code, and parsed features are available at:
[https://github.com/kadirkesgin/xai-scada/tree/main/pcap](https://github.com/kadirkesgin/xai-scada/tree/main/pcap)
