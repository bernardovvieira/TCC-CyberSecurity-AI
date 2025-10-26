#!/bin/bash
# run_all.sh - Execute complete pipeline
# Author: Bernardo Vivian Vieira

set -e  # Exit on error

echo "========================================"
echo "AI-IDS Pipeline - Complete Execution"
echo "========================================"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: Virtual environment not activated"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Parse arguments
SKIP_SANITY=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_IDS=false
SKIP_ADVERSARIAL=false
SKIP_ANALYSIS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sanity)
            SKIP_SANITY=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-ids)
            SKIP_IDS=true
            shift
            ;;
        --skip-adversarial)
            SKIP_ADVERSARIAL=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --help)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-sanity        Skip environment validation"
            echo "  --skip-preprocess    Skip data preprocessing"
            echo "  --skip-train         Skip model training"
            echo "  --skip-ids           Skip IDS integration"
            echo "  --skip-adversarial   Skip adversarial attacks"
            echo "  --skip-analysis      Skip complementary analysis"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Script 00: Sanity Check
if [ "$SKIP_SANITY" = false ]; then
    echo ""
    echo "========================================"
    echo "Step 0: Environment Validation"
    echo "========================================"
    python3 00_sanity_check.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Sanity check failed"
        exit 1
    fi
fi

# Script 01: Preprocessing
if [ "$SKIP_PREPROCESS" = false ]; then
    echo ""
    echo "========================================"
    echo "Step 1: Data Preprocessing"
    echo "========================================"
    python3 01_preprocessing.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed"
        exit 1
    fi
fi

# Script 02: Train and Evaluate
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "========================================"
    echo "Step 2: Model Training and Evaluation"
    echo "========================================"
    python3 02_train_evaluate.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed"
        exit 1
    fi
fi

# Script 03: IDS Integration
if [ "$SKIP_IDS" = false ]; then
    echo ""
    echo "========================================"
    echo "Step 3: IDS Integration (Suricata)"
    echo "========================================"
    if command -v suricata &> /dev/null; then
        python3 03_ids_integration.py
        if [ $? -ne 0 ]; then
            echo "WARNING: IDS integration failed (non-critical)"
        fi
    else
        echo "WARNING: Suricata not found, skipping IDS integration"
        echo "Install with: sudo apt-get install suricata"
    fi
fi

# Script 04: Adversarial Attacks
if [ "$SKIP_ADVERSARIAL" = false ]; then
    echo ""
    echo "========================================"
    echo "Step 4: Adversarial Attacks Analysis"
    echo "========================================"
    python3 04_adversarial.py
    if [ $? -ne 0 ]; then
        echo "WARNING: Adversarial analysis failed (non-critical)"
    fi
fi

# Script 05: Complementary Analysis
if [ "$SKIP_ANALYSIS" = false ]; then
    echo ""
    echo "========================================"
    echo "Step 5: Complementary Analysis"
    echo "========================================"
    python3 05_analysis.py
    if [ $? -ne 0 ]; then
        echo "WARNING: Complementary analysis failed (non-critical)"
    fi
fi

echo ""
echo "========================================"
echo "Pipeline Completed Successfully"
echo "========================================"
echo ""
echo "Results available in:"
echo "  - models/     (trained models)"
echo "  - results/    (metrics and plots)"
echo "  - runs/       (IDS outputs)"
echo ""