#!/usr/bin/env python3
"""
Script 05: Análises Complementares

Descrição:
    Unifica todas as análises complementares do pipeline, incluindo:
    1. Benchmark de latência (inferência e IDS)
    2. Curvas ROC e Precision-Recall
    3. Testes estatísticos (McNemar)
    4. Análise de importância de features
    5. Análise de erros de classificação
    6. Análise temporal de alertas
    7. Custo computacional

Funcionalidades:
    - Medição de latência e throughput
    - Geração de curvas ROC/PR macro-averaged
    - Validação estatística de diferenças entre modelos
    - Identificação de features mais importantes
    - Análise de padrões de erro e confusões
    - Distribuição temporal de alertas
    - Análise de consumo de recursos

Saída:
    results/latency_*.csv, results/roc_*.png, results/pr_*.png,
    results/auc_macro.csv, results/statistical_tests.csv,
    results/feature_importance_*.csv, results/error_analysis_*.csv,
    results/temporal_*.png, results/computational_cost.csv

Uso:
    python3 05_analysis.py
    python3 05_analysis.py --skip-temporal
    python3 05_analysis.py --skip-statistical

Autor: Bernardo Vivian Vieira
Data: Outubro 2025
"""

import os
import sys
import time
import logging
import argparse
import datetime
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(level="INFO"):
    """Configura sistema de logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s] %(message)s"
    )


def load_config():
    """Carrega configuração do arquivo YAML."""
    if not os.path.exists("config.yaml"):
        logging.error("Arquivo config.yaml não encontrado")
        sys.exit(1)
    
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Análises complementares do pipeline"
    )
    
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Pula benchmark de latência"
    )
    
    parser.add_argument(
        "--skip-roc",
        action="store_true",
        help="Pula geração de curvas ROC/PR"
    )
    
    parser.add_argument(
        "--skip-statistical",
        action="store_true",
        help="Pula testes estatísticos"
    )
    
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Pula análise de features"
    )
    
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Pula análise de erros"
    )
    
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Pula análise temporal"
    )
    
    parser.add_argument(
        "--skip-cost",
        action="store_true",
        help="Pula análise de custo computacional"
    )
    
    return parser.parse_args()


# =============================================================================
# MÓDULO 1: BENCHMARK DE LATÊNCIA
# =============================================================================

def measure_ia_latency(models_dir, results_dir):
    """
    Mede latência de inferência dos modelos de IA.
    
    Args:
        models_dir: Diretório contendo modelos treinados
        results_dir: Diretório com splits de dados
    
    Returns:
        DataFrame com métricas de latência
    """
    logging.info("Iniciando benchmark de latência de IA")
    
    splits_path = os.path.join(results_dir, "train_test_splits.pkl")
    if not os.path.exists(splits_path):
        logging.error("Splits não encontrados. Execute script 02 primeiro")
        return pd.DataFrame()
    
    _, X_test, _, _ = joblib.load(splits_path)
    logging.info(f"Amostras de teste carregadas: {len(X_test):,}")
    
    models_to_test = ["DecisionTree", "RandomForest", "MLP"]
    results = []
    
    for model_name in models_to_test:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            logging.warning(f"Modelo não encontrado: {model_path}")
            continue
        
        logging.info(f"Testando {model_name}")
        
        clf = joblib.load(model_path)
        
        # Warmup para evitar overhead inicial
        _ = clf.predict(X_test[:100])
        
        # Medição de latência
        n_runs = 5
        times = []
        
        for run in range(n_runs):
            t0 = time.perf_counter()
            _ = clf.predict(X_test)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        per_sample_ms = (mean_time / len(X_test)) * 1000
        throughput = len(X_test) / mean_time
        
        results.append({
            "Model": model_name,
            "Total_time_s": mean_time,
            "Std_time_s": std_time,
            "Infer_ms_per_sample": per_sample_ms,
            "Throughput_samples_per_s": throughput
        })
        
        logging.info(f"  Tempo total: {mean_time:.4f}s +/- {std_time:.4f}s")
        logging.info(f"  Por amostra: {per_sample_ms:.4f}ms")
        logging.info(f"  Throughput: {throughput:.0f} amostras/s")
    
    return pd.DataFrame(results)


def parse_timestamp(ts):
    """
    Parse timestamp de eve.json do Suricata.
    
    Args:
        ts: String com timestamp
    
    Returns:
        Objeto datetime ou None
    """
    try:
        ts = ts.replace("Z", "+0000")
        return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
    except:
        try:
            return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
        except:
            return None


def measure_ids_window(runs_dir):
    """
    Mede janela temporal dos alertas do IDS.
    
    Args:
        runs_dir: Diretório com execuções do Suricata
    
    Returns:
        DataFrame com janelas temporais por PCAP
    """
    logging.info("Medindo janela temporal do IDS")
    
    if not os.path.exists(runs_dir):
        logging.warning(f"Diretório de runs não existe: {runs_dir}")
        return pd.DataFrame()
    
    results = []
    
    for pcap_name in sorted(os.listdir(runs_dir)):
        pcap_dir = os.path.join(runs_dir, pcap_name)
        
        if not os.path.isdir(pcap_dir):
            continue
        
        csv_alert = None
        for f in os.listdir(pcap_dir):
            if f.endswith("_alerts.csv"):
                csv_alert = os.path.join(pcap_dir, f)
                break
        
        if not csv_alert or not os.path.exists(csv_alert):
            results.append({
                "pcap": pcap_name,
                "alert_window_s": None,
                "n_alerts": 0,
                "alerts_per_second": None
            })
            continue
        
        try:
            df = pd.read_csv(csv_alert, usecols=["timestamp"], 
                           nrows=100000, low_memory=False)
            
            if df.empty:
                results.append({
                    "pcap": pcap_name,
                    "alert_window_s": 0,
                    "n_alerts": 0,
                    "alerts_per_second": 0
                })
                continue
            
            timestamps = []
            for ts in df["timestamp"].dropna().head(100000):
                dt = parse_timestamp(str(ts))
                if dt:
                    timestamps.append(dt)
            
            n_alerts = len(df)
            
            if timestamps and len(timestamps) >= 2:
                window_s = (max(timestamps) - min(timestamps)).total_seconds()
                alerts_per_s = n_alerts / window_s if window_s > 0 else 0
            else:
                window_s = None
                alerts_per_s = None
            
            results.append({
                "pcap": pcap_name,
                "alert_window_s": window_s,
                "n_alerts": n_alerts,
                "alerts_per_second": alerts_per_s
            })
            
        except Exception as e:
            logging.warning(f"Erro ao processar {pcap_name}: {e}")
            results.append({
                "pcap": pcap_name,
                "alert_window_s": None,
                "n_alerts": 0,
                "alerts_per_second": None
            })
    
    return pd.DataFrame(results)


def plot_latency_comparison(ia_df, results_dir):
    """
    Plota comparação de latências entre modelos.
    
    Args:
        ia_df: DataFrame com latências de IA
        results_dir: Diretório de saída
    """
    if ia_df.empty:
        logging.warning("Sem dados de IA para plotar")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ia_df["Model"].values
    latencies = ia_df["Infer_ms_per_sample"].values
    
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax1.bar(models, latencies, color=colors[:len(models)], alpha=0.8)
    
    ax1.set_ylabel("Latencia (ms/amostra)")
    ax1.set_title("Latencia de Inferencia - Modelos de IA")
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    if "Throughput_samples_per_s" in ia_df.columns:
        throughputs = ia_df["Throughput_samples_per_s"].values
        
        bars2 = ax2.bar(models, throughputs, color=colors[:len(models)], alpha=0.8)
        
        ax2.set_ylabel("Amostras/segundo")
        ax2.set_title("Throughput - Modelos de IA")
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    img_path = os.path.join(results_dir, "latency_comparison.png")
    plt.savefig(img_path, dpi=160)
    plt.close()
    
    logging.info(f"Grafico de latencia salvo: {img_path}")


# =============================================================================
# MÓDULO 2: CURVAS ROC E PRECISION-RECALL
# =============================================================================

def plot_roc_macro(model_name, y_true, proba, class_names, results_dir):
    """
    Plota curva ROC macro-averaged.
    
    Args:
        model_name: Nome do modelo
        y_true: Labels verdadeiros
        proba: Probabilidades preditas
        class_names: Nomes das classes
        results_dir: Diretório de saída
    
    Returns:
        AUC macro
    """
    Y = label_binarize(y_true, classes=range(len(class_names)))
    
    fpr_list = []
    tpr_list = []
    
    for i in range(Y.shape[1]):
        fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(Y.shape[1]):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    
    mean_tpr /= Y.shape[1]
    roc_auc_macro = auc(all_fpr, mean_tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(all_fpr, mean_tpr, 
            label=f'ROC Macro (AUC = {roc_auc_macro:.3f})',
            color='steelblue', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Baseline')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC (Macro) - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(results_dir, f"roc_macro_{model_name}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    return roc_auc_macro


def plot_pr_macro(model_name, y_true, proba, class_names, results_dir):
    """
    Plota curva Precision-Recall macro-averaged.
    
    Args:
        model_name: Nome do modelo
        y_true: Labels verdadeiros
        proba: Probabilidades preditas
        class_names: Nomes das classes
        results_dir: Diretório de saída
    
    Returns:
        Average Precision macro
    """
    Y = label_binarize(y_true, classes=range(len(class_names)))
    
    ap_list = []
    
    for i in range(Y.shape[1]):
        ap = average_precision_score(Y[:, i], proba[:, i])
        ap_list.append(ap)
    
    ap_macro = np.mean(ap_list)
    
    plt.figure(figsize=(8, 6))
    
    for i in range(min(5, Y.shape[1])):
        precision, recall, _ = precision_recall_curve(Y[:, i], proba[:, i])
        plt.plot(recall, precision, 
                lw=1, alpha=0.6,
                label=f'{class_names[i][:20]}... (AP={ap_list[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curvas Precision-Recall - {model_name}\nMacro AP = {ap_macro:.3f}')
    plt.legend(loc='best', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(results_dir, f"pr_macro_{model_name}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    return ap_macro


def plot_roc_comparison(results_df, results_dir):
    """
    Plota comparação de AUC entre modelos.
    
    Args:
        results_df: DataFrame com resultados
        results_dir: Diretório de saída
    """
    if results_df.empty:
        return
    
    plt.figure(figsize=(8, 6))
    
    colors = ['steelblue', 'coral', 'seagreen']
    
    for idx, row in results_df.iterrows():
        model_name = row['Model']
        roc_auc = row['ROC_AUC_macro']
        
        plt.scatter(idx, roc_auc, s=200, alpha=0.7, 
                   color=colors[idx % len(colors)],
                   label=f'{model_name} (AUC={roc_auc:.3f})')
    
    plt.ylim([0.0, 1.05])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('ROC AUC Score (Macro)')
    plt.title('Comparacao de ROC AUC - Todos os Modelos')
    plt.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks([])
    plt.tight_layout()
    
    out_path = os.path.join(results_dir, "roc_comparison.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    logging.info(f"Comparacao ROC salva: {out_path}")


def generate_roc_pr_curves(models_dir, results_dir):
    """
    Gera curvas ROC e PR para todos os modelos.
    
    Args:
        models_dir: Diretório com modelos
        results_dir: Diretório de resultados
    
    Returns:
        DataFrame com métricas AUC
    """
    logging.info("Gerando curvas ROC e PR")
    
    splits_path = os.path.join(results_dir, "train_test_splits.pkl")
    if not os.path.exists(splits_path):
        logging.error("Splits não encontrados")
        return pd.DataFrame()
    
    _, X_test, _, y_test = joblib.load(splits_path)
    
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    le = joblib.load(le_path)
    class_names = list(le.classes_)
    
    models_to_plot = ["DecisionTree", "RandomForest", "MLP"]
    results = []
    
    for model_name in models_to_plot:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            logging.warning(f"Modelo não encontrado: {model_path}")
            continue
        
        logging.info(f"Processando {model_name}")
        
        clf = joblib.load(model_path)
        
        if not hasattr(clf, 'predict_proba'):
            logging.warning(f"{model_name} sem predict_proba")
            continue
        
        proba = clf.predict_proba(X_test)
        
        roc_auc = plot_roc_macro(model_name, y_test, proba, class_names, results_dir)
        ap_macro = plot_pr_macro(model_name, y_test, proba, class_names, results_dir)
        
        results.append({
            "Model": model_name,
            "ROC_AUC_macro": roc_auc,
            "AP_macro": ap_macro
        })
        
        logging.info(f"  ROC AUC: {roc_auc:.3f}, AP: {ap_macro:.3f}")
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        plot_roc_comparison(results_df, results_dir)
    
    return results_df


# =============================================================================
# MÓDULO 3: TESTES ESTATÍSTICOS
# =============================================================================

def mcnemar_test(preds1, preds2, y_true):
    """
    Executa teste de McNemar para comparar dois classificadores.
    
    Args:
        preds1: Predições do modelo 1
        preds2: Predições do modelo 2
        y_true: Labels verdadeiros
    
    Returns:
        Tupla (estatística, p-valor)
    """
    n01 = np.sum((preds1 == y_true) & (preds2 != y_true))
    n10 = np.sum((preds1 != y_true) & (preds2 == y_true))
    
    statistic = ((n01 - n10) ** 2) / (n01 + n10) if (n01 + n10) > 0 else 0
    p_value = 1 - stats.chi2.cdf(statistic, 1)
    
    return statistic, p_value


def perform_statistical_tests(models_dir, results_dir):
    """
    Realiza testes estatísticos de McNemar entre modelos.
    
    Args:
        models_dir: Diretório com modelos
        results_dir: Diretório de resultados
    
    Returns:
        DataFrame com resultados dos testes
    """
    logging.info("Executando testes estatisticos")
    
    splits_path = os.path.join(results_dir, "train_test_splits.pkl")
    _, X_test, _, y_test = joblib.load(splits_path)
    
    models = ["DecisionTree", "RandomForest", "MLP"]
    predictions = {}
    
    for model_name in models:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            predictions[model_name] = model.predict(X_test)
    
    results = []
    
    for i, model1 in enumerate(models):
        if model1 not in predictions:
            continue
            
        for model2 in models[i+1:]:
            if model2 not in predictions:
                continue
            
            stat, p_val = mcnemar_test(
                predictions[model1], 
                predictions[model2], 
                y_test
            )
            
            results.append({
                "Model_1": model1,
                "Model_2": model2,
                "McNemar_Statistic": stat,
                "p_value": p_val,
                "Significant_at_0.05": "Sim" if p_val < 0.05 else "Nao"
            })
    
    return pd.DataFrame(results)


# =============================================================================
# MÓDULO 4: FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(models_dir, results_dir, top_n=20):
    """
    Analisa importância das features usando Random Forest.
    
    Args:
        models_dir: Diretório com modelos
        results_dir: Diretório de resultados
        top_n: Número de top features a plotar
    
    Returns:
        DataFrame com importâncias
    """
    logging.info("Analisando importancia de features")
    
    rf_path = os.path.join(models_dir, "RandomForest.pkl")
    if not os.path.exists(rf_path):
        logging.warning("Random Forest não encontrado")
        return pd.DataFrame()
    
    rf = joblib.load(rf_path)
    
    import json
    with open(os.path.join(models_dir, "features_list.json"), "r") as f:
        features = json.load(f)
    
    importances = rf.feature_importances_
    
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [features[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Features Mais Importantes (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    img_path = os.path.join(results_dir, "feature_importance_RF.png")
    plt.savefig(img_path, dpi=160)
    plt.close()
    
    logging.info(f"Feature importance salva: {img_path}")
    
    return df


# =============================================================================
# MÓDULO 5: ANÁLISE DE ERROS
# =============================================================================

def analyze_confusion_patterns(cm, class_names, model_name, output_dir):
    """
    Identifica pares de classes mais confundidos.
    
    Args:
        cm: Matriz de confusão
        class_names: Nomes das classes
        model_name: Nome do modelo
        output_dir: Diretório de saída
    
    Returns:
        DataFrame com padrões de confusão
    """
    n_classes = len(class_names)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'True_Class': class_names[i],
                    'Predicted_As': class_names[j],
                    'Count': int(cm[i, j]),
                    'Percentage': float(cm_normalized[i, j] * 100)
                })
    
    df = pd.DataFrame(confusion_pairs).sort_values('Count', ascending=False)
    
    if len(df) > 0:
        top10 = df.head(10)
        
        plt.figure(figsize=(12, 6))
        labels = [f"{row['True_Class'][:15]}\n-> {row['Predicted_As'][:15]}" 
                 for _, row in top10.iterrows()]
        
        plt.barh(range(len(top10)), top10['Count'], color='coral')
        plt.yticks(range(len(top10)), labels, fontsize=9)
        plt.xlabel('Numero de Erros')
        plt.title(f'{model_name}: Top 10 Confusoes Mais Frequentes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        img_path = os.path.join(output_dir, f"confusion_pairs_{model_name}.png")
        plt.savefig(img_path, dpi=160)
        plt.close()
    
    return df


def perform_error_analysis(models_dir, results_dir):
    """
    Realiza análise detalhada de erros para todos os modelos.
    
    Args:
        models_dir: Diretório com modelos
        results_dir: Diretório de resultados
    
    Returns:
        Dicionário com DataFrames de análise por modelo
    """
    logging.info("Analisando padroes de erro")
    
    splits_path = os.path.join(results_dir, "train_test_splits.pkl")
    _, X_test, _, y_test = joblib.load(splits_path)
    
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    le = joblib.load(le_path)
    class_names = list(le.classes_)
    
    results = {}
    
    for model_name in ["DecisionTree", "RandomForest", "MLP"]:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            continue
        
        logging.info(f"Analisando erros de {model_name}")
        
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        df = analyze_confusion_patterns(cm, class_names, model_name, results_dir)
        
        results[model_name] = df
        
        if len(df) > 0:
            logging.info(f"  Top confusao: {df.iloc[0]['True_Class']} -> "
                        f"{df.iloc[0]['Predicted_As']} ({df.iloc[0]['Count']} casos)")
    
    return results


# =============================================================================
# MÓDULO 6: ANÁLISE TEMPORAL
# =============================================================================

def analyze_temporal_distribution(runs_dir, results_dir):
    """
    Analisa distribuição temporal de alertas do Suricata.
    
    Args:
        runs_dir: Diretório com execuções do Suricata
        results_dir: Diretório de resultados
    
    Returns:
        Dicionário com distribuições horárias por PCAP
    """
    logging.info("Analisando distribuicao temporal de alertas")
    
    if not os.path.exists(runs_dir):
        logging.warning(f"Diretorio de runs nao existe: {runs_dir}")
        return {}
    
    temporal_data = {}
    
    for pcap_name in sorted(os.listdir(runs_dir)):
        pcap_dir = os.path.join(runs_dir, pcap_name)
        
        if not os.path.isdir(pcap_dir):
            continue
        
        csv_alert = None
        for f in os.listdir(pcap_dir):
            if f.endswith("_alerts.csv"):
                csv_alert = os.path.join(pcap_dir, f)
                break
        
        if not csv_alert or not os.path.exists(csv_alert):
            continue
        
        try:
            df = pd.read_csv(csv_alert)
            if 'timestamp' not in df.columns:
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df[df['timestamp'].notna()]
            
            if len(df) == 0:
                continue
            
            df['hour'] = df['timestamp'].dt.hour
            hourly = df.groupby('hour').size()
            
            temporal_data[pcap_name] = hourly
            
            plt.figure(figsize=(12, 4))
            hourly.plot(kind='bar', color='steelblue')
            plt.xlabel('Hora do Dia')
            plt.ylabel('Numero de Alertas')
            plt.title(f'Distribuicao Temporal - {pcap_name}')
            plt.tight_layout()
            
            img_path = os.path.join(results_dir, f"temporal_{pcap_name}.png")
            plt.savefig(img_path, dpi=160)
            plt.close()
            
            logging.info(f"  {pcap_name}: Pico em {hourly.idxmax()}h ({hourly.max()} alertas)")
            
        except Exception as e:
            logging.warning(f"Erro ao processar {pcap_name}: {e}")
    
    return temporal_data


# =============================================================================
# MÓDULO 7: CUSTO COMPUTACIONAL
# =============================================================================

def analyze_computational_cost(models_dir, results_dir):
    """
    Analisa custo computacional (tamanho e parâmetros dos modelos).
    
    Args:
        models_dir: Diretório com modelos
        results_dir: Diretório de resultados
    
    Returns:
        DataFrame com análise de custo
    """
    logging.info("Analisando custo computacional")
    
    results = []
    
    for model_name in ["DecisionTree", "RandomForest", "MLP"]:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            continue
        
        model = joblib.load(model_path)
        
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        n_params = getattr(model, 'n_features_in_', 'N/A')
        
        results.append({
            "Model": model_name,
            "Size_MB": round(size_mb, 2),
            "Parameters": n_params
        })
        
        logging.info(f"  {model_name}: {size_mb:.2f} MB")
    
    return pd.DataFrame(results)


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Função principal que executa todas as análises."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    logging.info("="*70)
    logging.info("SCRIPT 05: ANALISES COMPLEMENTARES")
    logging.info("="*70)
    
    args = parse_args()
    
    models_dir = config["paths"]["models_dir"]
    results_dir = config["paths"]["results_dir"]
    runs_dir = config["paths"]["runs_dir"]
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Módulo 1: Benchmark de Latência
    if not args.skip_latency:
        logging.info("\n" + "="*70)
        logging.info("MODULO 1: BENCHMARK DE LATENCIA")
        logging.info("="*70)
        
        ia_df = measure_ia_latency(models_dir, results_dir)
        if not ia_df.empty:
            ia_path = os.path.join(results_dir, "latency_ia.csv")
            ia_df.to_csv(ia_path, index=False)
            logging.info(f"Latencias de IA salvas: {ia_path}")
            
            plot_latency_comparison(ia_df, results_dir)
        
        ids_df = measure_ids_window(runs_dir)
        if not ids_df.empty:
            ids_path = os.path.join(results_dir, "latency_ids.csv")
            ids_df.to_csv(ids_path, index=False)
            logging.info(f"Janelas do IDS salvas: {ids_path}")
    
    # Módulo 2: Curvas ROC e PR
    if not args.skip_roc:
        logging.info("\n" + "="*70)
        logging.info("MODULO 2: CURVAS ROC E PRECISION-RECALL")
        logging.info("="*70)
        
        roc_df = generate_roc_pr_curves(models_dir, results_dir)
        if not roc_df.empty:
            auc_path = os.path.join(results_dir, "auc_macro.csv")
            roc_df.to_csv(auc_path, index=False)
            logging.info(f"Metricas AUC salvas: {auc_path}")
    
    # Módulo 3: Testes Estatísticos
    if not args.skip_statistical:
        logging.info("\n" + "="*70)
        logging.info("MODULO 3: TESTES ESTATISTICOS")
        logging.info("="*70)
        
        stat_df = perform_statistical_tests(models_dir, results_dir)
        if not stat_df.empty:
            stat_path = os.path.join(results_dir, "statistical_tests.csv")
            stat_df.to_csv(stat_path, index=False)
            logging.info(f"Testes estatisticos salvos: {stat_path}")
            
            logging.info("\nResultados dos testes de McNemar:")
            for _, row in stat_df.iterrows():
                logging.info(f"  {row['Model_1']} vs {row['Model_2']}: "
                           f"p={row['p_value']:.4f} "
                           f"({'significativo' if row['p_value'] < 0.05 else 'nao significativo'})")
    
    # Módulo 4: Feature Importance
    if not args.skip_features:
        logging.info("\n" + "="*70)
        logging.info("MODULO 4: IMPORTANCIA DE FEATURES")
        logging.info("="*70)
        
        feat_df = analyze_feature_importance(models_dir, results_dir)
        if not feat_df.empty:
            feat_path = os.path.join(results_dir, "feature_importance_RF.csv")
            feat_df.to_csv(feat_path, index=False)
            logging.info(f"Feature importance salva: {feat_path}")
    
    # Módulo 5: Análise de Erros
    if not args.skip_errors:
        logging.info("\n" + "="*70)
        logging.info("MODULO 5: ANALISE DE ERROS")
        logging.info("="*70)
        
        error_results = perform_error_analysis(models_dir, results_dir)
        for model_name, df in error_results.items():
            if not df.empty:
                error_path = os.path.join(results_dir, f"error_analysis_{model_name}.csv")
                df.to_csv(error_path, index=False)
                logging.info(f"Analise de erros {model_name} salva: {error_path}")
    
    # Módulo 6: Análise Temporal
    if not args.skip_temporal:
        logging.info("\n" + "="*70)
        logging.info("MODULO 6: ANALISE TEMPORAL")
        logging.info("="*70)
        
        temporal_data = analyze_temporal_distribution(runs_dir, results_dir)
        if temporal_data:
            logging.info(f"Distribuicoes temporais geradas para {len(temporal_data)} PCAPs")
    
    # Módulo 7: Custo Computacional
    if not args.skip_cost:
        logging.info("\n" + "="*70)
        logging.info("MODULO 7: CUSTO COMPUTACIONAL")
        logging.info("="*70)
        
        cost_df = analyze_computational_cost(models_dir, results_dir)
        if not cost_df.empty:
            cost_path = os.path.join(results_dir, "computational_cost.csv")
            cost_df.to_csv(cost_path, index=False)
            logging.info(f"Custo computacional salvo: {cost_path}")
    
    logging.info("\n" + "="*70)
    logging.info("ANALISES COMPLEMENTARES CONCLUIDAS")
    logging.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())