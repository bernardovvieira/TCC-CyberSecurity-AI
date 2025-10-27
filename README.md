# AI-IDS: Artificial Intelligence Integration with Intrusion Detection Systems

**Author:** Bernardo Vivian Vieira  
**Email:** 179835@upf.br

---

## Installation

### 1. Download Dataset
Download CIC-IDS-2017 from: https://www.unb.ca/cic/datasets/ids-2017.html

Or use wget (Linux/Mac):
```bash
wget -r -np -nH --cut-dirs=3 -R "index.html*" \
    http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment
```bash
source venv/bin/activate  # Linux/Mac
```
or
```bash
venv\Scripts\activate  # Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install Suricata - Ubuntu/Debian (Optional - for IDS integration)
```bash
sudo apt update
sudo apt install suricata
sudo suricata-update
```

---

## Usage

### Run Complete Pipeline (Linux/Mac)
```bash
chmod +x run_all.sh
./run_all.sh
```

### Run Individual Scripts
```bash
python3 00_sanity_check.py
python3 01_preprocessing.py
python3 02_train_evaluate.py
python3 03_ids_integration.py
python3 04_adversarial.py
python3 05_analysis.py
```