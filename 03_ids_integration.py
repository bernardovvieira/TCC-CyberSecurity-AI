#!/usr/bin/env python3
"""
Script 03: Integração IDS (Suricata)

Descrição:
    Integração completa entre IDS baseado em regras (Suricata) e modelos de
    Machine Learning, incluindo processamento de PCAPs, análise de alertas e
    comparação de desempenho.

Uso:
    python3 03_ids_integration.py
    python3 03_ids_integration.py --skip-suricata
    python3 03_ids_integration.py --max-alerts-per-pcap 0

Autor: Bernardo Vivian Vieira
Data: Outubro 2025
"""

import argparse
import gc
import glob
import json
import logging
import os
import sys
import subprocess
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_logging(level="INFO"):
    """Configura sistema de logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s] %(message)s"
    )


def load_config():
    """Carrega arquivo de configuração YAML."""
    if not os.path.exists("config.yaml"):
        logging.error("Arquivo config.yaml não encontrado")
        sys.exit(1)
    
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def parse_args(config):
    """Processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Integração IDS otimizada"
    )
    
    parser.add_argument(
        "--skip-suricata",
        action="store_true",
        help="Pula processamento de PCAPs"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Tamanho dos chunks para processamento (padrão: 100000)"
    )
    
    parser.add_argument(
        "--max-alerts-per-pcap",
        type=int,
        default=0,
        help="Máximo de alertas por PCAP (0 = sem limite)"
    )
    
    return parser.parse_args()


# ============================================================================
# PARTE 1: PROCESSAMENTO SURICATA
# ============================================================================

def run_command(cmd, timeout=3600):
    """
    Executa comando do sistema com timeout.
    
    Args:
        cmd: Lista com comando
        timeout: Timeout em segundos (padrão: 1 hora)
    
    Returns:
        Tupla (código de retorno, saída)
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired:
        logging.error(f"Comando excedeu timeout de {timeout}s")
        return -1, ""
    except Exception as e:
        logging.error(f"Erro ao executar comando: {e}")
        return -1, ""


def process_pcap_with_suricata(pcap_path, output_dir, suricata_bin):
    """
    Processa PCAP com Suricata de forma segura.
    
    Args:
        pcap_path: Caminho do PCAP
        output_dir: Diretório de saída
        suricata_bin: Binário do Suricata
    
    Returns:
        Caminho do eve.json ou None
    """
    base_name = os.path.splitext(os.path.basename(pcap_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se já foi processado
    eve_path = os.path.join(output_dir, "eve.json")
    if os.path.exists(eve_path) and os.path.getsize(eve_path) > 0:
        logging.info(f"{base_name} já processado, pulando...")
        return eve_path
    
    # Verificar tamanho do PCAP
    pcap_size_mb = os.path.getsize(pcap_path) / (1024 * 1024)
    logging.info(f"Processando {base_name} ({pcap_size_mb:.1f} MB)...")
    
    # Ajustar timeout baseado no tamanho
    timeout = min(3600, max(300, int(pcap_size_mb * 10)))
    
    cmd = ["sudo", suricata_bin, "-r", pcap_path, "-l", output_dir]
    
    logging.info(f"Timeout configurado: {timeout}s")
    rc, output = run_command(cmd, timeout=timeout)
    
    if rc != 0:
        logging.warning(f"Suricata retornou código {rc} para {base_name}")
        if "out of memory" in output.lower():
            logging.error(f"Erro de memória ao processar {base_name}")
            return None
    
    if not os.path.exists(eve_path):
        logging.warning(f"eve.json não gerado para {base_name}")
        return None
    
    return eve_path


def parse_eve_to_csv_optimized(eve_path, output_csv, max_alerts=0):
    """
    Parse eve.json para CSV de forma memory-efficient.
    
    Args:
        eve_path: Caminho do eve.json
        output_csv: Caminho do CSV de saída
        max_alerts: Máximo de alertas a processar (0 = sem limite)
    
    Returns:
        Número de alertas processados
    """
    if not os.path.exists(eve_path):
        return 0
    
    alert_count = 0
    rows = []
    
    with open(eve_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if max_alerts > 0 and alert_count >= max_alerts:
                logging.info(f"Limite de {max_alerts} alertas atingido")
                break
            
            try:
                event = json.loads(line)
                
                if event.get("event_type") != "alert":
                    continue
                
                alert = event.get("alert", {})
                
                rows.append({
                    "timestamp": event.get("timestamp"),
                    "src_ip": event.get("src_ip"),
                    "dest_ip": event.get("dest_ip"),
                    "severity": alert.get("severity"),
                    "signature_id": alert.get("signature_id"),
                    "signature": alert.get("signature"),
                    "category": alert.get("category")
                })
                
                alert_count += 1
                
                if alert_count % 10000 == 0:
                    logging.info(f"  Processados {alert_count} alertas...")
                
            except json.JSONDecodeError:
                continue
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        del df
        gc.collect()
    
    return alert_count


# ============================================================================
# PARTE 2: ANÁLISE SIMPLIFICADA (SEM CORRELAÇÃO TEMPORAL PESADA)
# ============================================================================

def aggregate_suricata_alerts(runs_dir, results_dir):
    """
    Agrega alertas do Suricata por categoria.
    
    Args:
        runs_dir: Diretório com runs
        results_dir: Diretório de saída
    """
    logging.info("Agregando alertas do Suricata...")
    
    all_alerts = []
    
    for pcap_dir in os.listdir(runs_dir):
        pcap_path = os.path.join(runs_dir, pcap_dir)
        
        if not os.path.isdir(pcap_path):
            continue
        
        csv_path = None
        for f in os.listdir(pcap_path):
            if f.endswith("_alerts.csv"):
                csv_path = os.path.join(pcap_path, f)
                break
        
        if not csv_path or not os.path.exists(csv_path):
            continue
        
        try:
            df = pd.read_csv(csv_path, usecols=["category", "severity"])
            all_alerts.append(df)
        except Exception as e:
            logging.warning(f"Erro ao processar {pcap_dir}: {e}")
    
    if not all_alerts:
        logging.warning("Nenhum alerta encontrado")
        return
    
    combined = pd.concat(all_alerts, ignore_index=True)
    del all_alerts
    gc.collect()
    
    by_category = (combined.groupby("category", dropna=False)
                   .size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False))
    
    output_path = os.path.join(results_dir, "suricata_summary_by_category.csv")
    by_category.to_csv(output_path, index=False)
    
    logging.info(f"Agregação salva: {output_path}")
    logging.info(f"Total de alertas: {by_category['count'].sum():,}")


def calculate_basic_ids_metrics(runs_dir, results_dir):
    """
    Calcula métricas básicas do IDS sem correlação temporal pesada.
    
    Args:
        runs_dir: Diretório com runs
        results_dir: Diretório de resultados
    
    Returns:
        DataFrame com métricas básicas
    """
    logging.info("Calculando métricas básicas do IDS...")
    
    results = []
    
    for pcap_dir in sorted(os.listdir(runs_dir)):
        pcap_path = os.path.join(runs_dir, pcap_dir)
        
        if not os.path.isdir(pcap_path):
            continue
        
        csv_path = None
        for f in os.listdir(pcap_path):
            if f.endswith("_alerts.csv"):
                csv_path = os.path.join(pcap_path, f)
                break
        
        if not csv_path:
            continue
        
        try:
            df = pd.read_csv(csv_path)
            
            total_alerts = len(df)
            unique_signatures = df["signature"].nunique()
            unique_categories = df["category"].nunique()
            
            severity_dist = df["severity"].value_counts().to_dict()
            
            results.append({
                "pcap": pcap_dir,
                "total_alerts": total_alerts,
                "unique_signatures": unique_signatures,
                "unique_categories": unique_categories,
                "severity_1": severity_dist.get(1, 0),
                "severity_2": severity_dist.get(2, 0),
                "severity_3": severity_dist.get(3, 0)
            })
            
            logging.info(f"{pcap_dir}: {total_alerts} alertas")
            
        except Exception as e:
            logging.warning(f"Erro ao processar {pcap_dir}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(results_dir, "suricata_basic_metrics.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Métricas básicas salvas: {output_path}")
        return df
    
    return pd.DataFrame()


# ============================================================================
# PARTE 3: COMPARAÇÃO COM MÉTRICAS DE IA
# ============================================================================

def create_simple_comparison(results_dir):
    """
    Cria comparação simples entre IA e IDS.
    
    Args:
        results_dir: Diretório de resultados
    """
    logging.info("Criando comparação IA vs IDS...")
    
    ia_metrics_path = os.path.join(results_dir, "overall_metrics.csv")
    
    if not os.path.exists(ia_metrics_path):
        logging.warning("Métricas de IA não encontradas")
        return
    
    ia_df = pd.read_csv(ia_metrics_path)
    
    comparison_rows = []
    
    for _, row in ia_df.iterrows():
        comparison_rows.append({
            "Fonte": "IA",
            "Modelo": row["Model"],
            "Accuracy": round(float(row["Accuracy"]) * 100, 2),
            "TPR": round(float(row.get("Macro_Recall(TPR)", 0)) * 100, 2),
            "FPR": round(float(row.get("Macro_FPR", 0)) * 100, 4),
            "F1": round(float(row.get("Macro_F1", 0)) * 100, 2),
            "Latency_ms": round(float(row.get("Infer_Time_ms", 0)), 4)
        })
    
    comparison_rows.append({
        "Fonte": "IDS",
        "Modelo": "Suricata",
        "Accuracy": "N/A",
        "TPR": "Varia por regra",
        "FPR": "Varia por regra",
        "F1": "N/A",
        "Latency_ms": "Depende de tráfego"
    })
    
    comp_df = pd.DataFrame(comparison_rows)
    
    output_path = os.path.join(results_dir, "ia_vs_ids_comparison_simple.csv")
    comp_df.to_csv(output_path, index=False)
    
    logging.info(f"Comparação salva: {output_path}")
    
    with open(os.path.join(results_dir, "ia_vs_ids_comparison_simple.tex"), "w") as f:
        f.write(comp_df.to_latex(
            index=False,
            caption="Comparacao IA vs IDS",
            label="tab:ia_vs_ids_simple",
            escape=False
        ))


def plot_simple_comparison(results_dir):
    """Plota comparação visual simples."""
    ia_metrics_path = os.path.join(results_dir, "overall_metrics.csv")
    
    if not os.path.exists(ia_metrics_path):
        return
    
    ia_df = pd.read_csv(ia_metrics_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ia_df["Model"]
    accuracy = ia_df["Accuracy"] * 100
    
    colors = ['steelblue', 'coral', 'seagreen']
    bars1 = ax1.bar(range(len(models)), accuracy, color=colors, alpha=0.8)
    
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Acuracia dos Modelos de IA")
    ax1.set_ylim([95, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    latency = ia_df["Infer_Time_ms"]
    
    bars2 = ax2.bar(range(len(models)), latency, color=colors, alpha=0.8)
    
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel("Latência (ms/amostra)")
    ax2.set_title("Latencia de Inferencia")
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, "ia_performance_overview.png")
    plt.savefig(output_path, dpi=160)
    plt.close()
    
    logging.info(f"Gráfico salvo: {output_path}")


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    logging.info("="*70)
    logging.info("SCRIPT 03: INTEGRACAO IDS")
    logging.info("="*70)
    
    args = parse_args(config)
    
    models_dir = config["paths"]["models_dir"]
    results_dir = config["paths"]["results_dir"]
    runs_dir = config["paths"]["runs_dir"]
    pcap_dir = config["paths"]["pcap_dir"]
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    
    # PARTE 1: Processamento com Suricata
    if not args.skip_suricata:
        logging.info("\n" + "="*70)
        logging.info("PROCESSAMENTO DE PCAPS COM SURICATA")
        logging.info("="*70)
        
        pcap_files = sorted(
            glob.glob(os.path.join(pcap_dir, "*.pcap")) +
            glob.glob(os.path.join(pcap_dir, "*.pcapng"))
        )
        
        if not pcap_files:
            logging.warning(f"Nenhum PCAP encontrado em {pcap_dir}")
        else:
            logging.info(f"Encontrados {len(pcap_files)} arquivos PCAP")
            logging.info("IMPORTANTE: Processamento pode levar vários minutos por PCAP")
            
            for idx, pcap_path in enumerate(pcap_files, 1):
                logging.info(f"\n[{idx}/{len(pcap_files)}] Iniciando processamento...")
                
                base_name = os.path.splitext(os.path.basename(pcap_path))[0]
                output_dir = os.path.join(runs_dir, base_name)
                
                try:
                    eve_path = process_pcap_with_suricata(
                        pcap_path,
                        output_dir,
                        config["suricata"]["binary"]
                    )
                    
                    if eve_path:
                        csv_path = os.path.join(output_dir, f"{base_name}_alerts.csv")
                        alert_count = parse_eve_to_csv_optimized(
                            eve_path,
                            csv_path,
                            args.max_alerts_per_pcap
                        )
                        logging.info(f"Sucesso: {alert_count} alertas extraídos")
                    else:
                        logging.warning(f"Falha ao processar {base_name}")
                
                except KeyboardInterrupt:
                    logging.warning("\nProcessamento interrompido pelo usuário")
                    break
                except Exception as e:
                    logging.error(f"Erro ao processar {base_name}: {e}")
                    continue
                finally:
                    # Limpeza agressiva de memória após cada PCAP
                    gc.collect()
                    logging.info(f"Memória liberada após {base_name}")
            
            logging.info(f"\nProcessamento de PCAPs concluído")
    else:
        logging.info("\nProcessamento de PCAPs pulado (--skip-suricata)")
    
    # PARTE 2: Análise de Alertas
    logging.info("\n" + "="*70)
    logging.info("ANALISE DE ALERTAS DO SURICATA")
    logging.info("="*70)
    
    aggregate_suricata_alerts(runs_dir, results_dir)
    calculate_basic_ids_metrics(runs_dir, results_dir)
    
    # PARTE 3: Comparação com IA
    logging.info("\n" + "="*70)
    logging.info("COMPARACAO IA VS IDS")
    logging.info("="*70)
    
    create_simple_comparison(results_dir)
    plot_simple_comparison(results_dir)
    
    logging.info("\n" + "="*70)
    logging.info("SCRIPT 03 CONCLUIDO")
    logging.info("="*70)
    logging.info("\nArquivos gerados:")
    logging.info("  - suricata_summary_by_category.csv")
    logging.info("  - suricata_basic_metrics.csv")
    logging.info("  - ia_vs_ids_comparison_simple.csv")
    logging.info("  - ia_performance_overview.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())