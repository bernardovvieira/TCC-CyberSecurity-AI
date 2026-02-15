# AI-IDS: Artificial Intelligence Integration with Intrusion Detection Systems

**Author:** Bernardo Vivian Vieira  
**Email:** 179835@upf.br

Implementation and comparative analysis of ML models (Decision Tree, Random Forest, MLP) integrated with Suricata IDS using the CIC-IDS-2017 dataset. Includes preprocessing, training, IDS integration, adversarial attacks (FGSM, PGD, C&W L2), and complementary analysis (ROC/PR, McNemar, feature importance).

---

## Requirements

- Ubuntu/Debian Linux
- Python 3.8+
- 60GB+ free disk space (dataset)

---

## Installation

### 1. System Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 2. Download Dataset

Download CIC-IDS-2017 from: https://www.unb.ca/cic/datasets/ids-2017.html

Or use wget:
```bash
wget -r -np -nH --cut-dirs=3 -R "index.html*" -c -nc \
    http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/
```

### 3. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configuration

The project uses `config.yaml` in the repository root for paths, model hyperparameters, and Suricata settings. Adjust `paths.data_dir`, `paths.csv_dir`, and `paths.pcap_dir` if your CIC-IDS-2017 layout differs (e.g. after extracting the dataset).

### 6. Install IDS Tools (Optional)

```bash
sudo apt install suricata tcpreplay tshark
sudo suricata-update
```

---

## Usage

### Run Complete Pipeline

```bash
chmod +x run_all.sh
./run_all.sh
```

### Run Individual Scripts

```bash
python3 00_sanity_check.py       # Validate environment
python3 01_preprocessing.py      # Process dataset
python3 02_train_evaluate.py     # Train models
python3 03_ids_integration.py    # IDS integration (requires Suricata)
python3 04_adversarial.py        # Adversarial attacks
python3 05_analysis.py           # Complementary analysis
```

---

## Output

Results will be generated in:
- `models/` - Trained models (.pkl files)
- `results/` - Metrics, plots, and reports
- `runs/` - IDS execution logs

---

## Citation

If you use this work in academic or applied research, please cite:

```bibtex
@software{vieira2025_ai_ids,
  author = {Vieira, Bernardo Vivian},
  title = {TCC-CyberSecurity-AI: Implementation and comparative analysis of ML models integrated into Suricata IDS using CIC-IDS-2017},
  year = {2025},
  url = {https://github.com/bernardovvieira/TCC-CyberSecurity-AI},
  note = {Undergraduate thesis (TCC)}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).