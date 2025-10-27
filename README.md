# AI-IDS: Artificial Intelligence Integration with Intrusion Detection Systems

**Author:** Bernardo Vivian Vieira  
**Email:** 179835@upf.br

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

### 5. Install IDS Tools (Optional)

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