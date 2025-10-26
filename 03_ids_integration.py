#!/usr/bin/env python3
"""
Script 03: Integração e Análise IDS

Descrição:
    Script unificado que realiza processamento de PCAPs com Suricata,
    integração dos modelos de IA com alertas do IDS e análise comparativa
    de desempenho entre as abordagens.

Funcionalidades:
    - Processamento de PCAPs com Suricata (geração de eve.json)
    - Extração de features do CIC-IDS-2017 a partir de logs Suricata
    - Aplicação de modelos de IA sobre dados do IDS
    - Correlação temporal entre alertas e ground truth
    - Cálculo de métricas TPR/FPR para IDS
    - Comparação estatística IA vs IDS
    - Análise de vulnerabilidades por feature

Saída:
    - suricata_runs/<pcap>/eve.json
    - results/suricata_summary_by_category.csv
    - results/ia_ids_integration_<modelo>.csv
    - results/suricata_vs_groundtruth.csv
    - results/ia_vs_ids_comparison.csv
    - results/feature_extraction_quality.csv
    - results/unmapped_sids_summary.csv

Uso:
    python3 03_ids_integration.py
    python3 03_ids_integration.py --skip-suricata
    python3 03_ids_integration.py --temporal-window 10

Autor: Bernardo Vivian Vieira
Data: Outubro 2025
"""

import argparse
import glob
import json
import logging
import os
import sys
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
        description="Integração e análise de IDS com modelos de IA"
    )
    
    parser.add_argument(
        "--skip-suricata",
        action="store_true",
        help="Pula processamento de PCAPs com Suricata"
    )
    
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Janela temporal para correlação em segundos (padrão: 5)"
    )
    
    parser.add_argument(
        "--pcap-dir",
        default=config["paths"]["pcap_dir"],
        help="Diretório contendo arquivos PCAP"
    )
    
    parser.add_argument(
        "--runs-dir",
        default=config["paths"]["runs_dir"],
        help="Diretório de saída para execuções do Suricata"
    )
    
    return parser.parse_args()


# ============================================================================
# PARTE 1: PROCESSAMENTO COM SURICATA
# ============================================================================

def run_command(cmd):
    """
    Executa comando do sistema operacional.
    
    Args:
        cmd: Lista com comando e argumentos
        
    Returns:
        Tupla (código de retorno, saída padrão)
    """
    logging.debug(f"Executando comando: {' '.join(cmd)}")
    
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    return proc.returncode, proc.stdout


def update_suricata_rules(skip_update):
    """
    Atualiza regras do Suricata se necessário.
    
    Args:
        skip_update: Se True, pula atualização
    """
    if skip_update:
        logging.info("Atualização de regras desabilitada")
        return
    
    logging.info("Atualizando regras do Suricata...")
    
    for cmd in (["sudo", "suricata-update"], ["suricata-update"]):
        try:
            rc, output = run_command(cmd)
            if rc == 0:
                logging.info("Regras atualizadas com sucesso")
                return
        except Exception as e:
            logging.debug(f"Tentativa de atualização falhou: {e}")
    
    logging.warning("Não foi possível atualizar regras automaticamente")


def process_pcap_with_suricata(pcap_path, output_dir, suricata_bin):
    """
    Processa arquivo PCAP com Suricata.
    
    Args:
        pcap_path: Caminho do arquivo PCAP
        output_dir: Diretório de saída
        suricata_bin: Caminho do binário Suricata
        
    Returns:
        Caminho do arquivo eve.json gerado
    """
    base_name = os.path.splitext(os.path.basename(pcap_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "sudo",
        suricata_bin,
        "-r", pcap_path,
        "-l", output_dir
    ]
    
    logging.info(f"Processando {base_name}...")
    rc, stdout = run_command(cmd)
    
    if rc != 0:
        logging.warning(f"Suricata retornou código {rc} para {base_name}")
    
    eve_path = os.path.join(output_dir, "eve.json")
    
    if not os.path.exists(eve_path):
        logging.warning(f"Arquivo eve.json não gerado para {base_name}")
    
    return eve_path


def parse_eve_to_dataframe(eve_path):
    """
    Converte eve.json para DataFrame de alertas.
    
    Args:
        eve_path: Caminho do arquivo eve.json
        
    Returns:
        DataFrame com alertas
    """
    if not os.path.exists(eve_path):
        logging.warning(f"Arquivo eve.json não encontrado: {eve_path}")
        return pd.DataFrame()
    
    rows = []
    
    with open(eve_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if event.get("event_type") != "alert":
                continue
            
            alert = event.get("alert", {})
            
            rows.append({
                "timestamp": event.get("timestamp"),
                "src_ip": event.get("src_ip"),
                "src_port": event.get("src_port"),
                "dest_ip": event.get("dest_ip"),
                "dest_port": event.get("dest_port"),
                "proto": event.get("proto"),
                "severity": alert.get("severity"),
                "signature_id": alert.get("signature_id"),
                "signature": alert.get("signature"),
                "category": alert.get("category")
            })
    
    return pd.DataFrame(rows)


def aggregate_alerts_by_category(csv_paths, output_path):
    """
    Agrega alertas por categoria.
    
    Args:
        csv_paths: Lista de caminhos de CSVs com alertas
        output_path: Caminho do arquivo de saída
    """
    frames = []
    
    for csv_path in csv_paths:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logging.warning(f"Erro ao processar {csv_path}: {e}")
    
    if not frames:
        logging.warning("Nenhum alerta válido para agregação")
        pd.DataFrame(columns=["category", "count"]).to_csv(output_path, index=False)
        return
    
    combined = pd.concat(frames, ignore_index=True)
    
    by_category = (combined.groupby("category", dropna=False)
                   .size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False))
    
    by_category.to_csv(output_path, index=False)
    logging.info(f"Agregação por categoria salva em: {output_path}")


# ============================================================================
# PARTE 2: MAPEAMENTO SURICATA → CIC-IDS-2017
# ============================================================================

SURICATA_SID_TO_CICIDS = {
    # DDoS / DoS
    "2100498": "DDoS", "2013028": "DDoS", "2001569": "DDoS",
    "2001331": "DDoS", "2100366": "DDoS", "2003194": "DDoS",
    "2024218": "DDoS", "2024220": "DDoS", "2025360": "DDoS",
    "2027512": "DDoS", "2100494": "DDoS", "2210014": "DDoS",
    
    # SSH Brute Force
    "2001219": "SSH-Patator", "2210000": "SSH-Patator",
    "2210001": "SSH-Patator", "2003068": "SSH-Patator",
    "2013926": "SSH-Patator", "2001251": "SSH-Patator",
    "2403322": "SSH-Patator", "2403324": "SSH-Patator",
    
    # FTP Brute Force
    "2010935": "FTP-Patator", "2010936": "FTP-Patator",
    "2010937": "FTP-Patator", "2221001": "FTP-Patator",
    "2010939": "FTP-Patator", "2010940": "FTP-Patator",
    "2221002": "FTP-Patator",
    
    # Web Attacks
    "2012887": "Web Attack", "2012889": "Web Attack",
    "2013504": "Web Attack", "2014466": "Web Attack",
    "2012648": "Web Attack", "2014149": "Web Attack",
    "2210050": "Web Attack", "2013028": "Web Attack",
    "2019105": "Web Attack", "2210051": "Web Attack",
    "2016538": "Web Attack", "2016540": "Web Attack",
    "2210052": "Web Attack", "2013357": "Web Attack",
    "2018088": "Web Attack",
    
    # PortScan
    "2000356": "PortScan", "2001569": "PortScan",
    "2009187": "PortScan", "2402000": "PortScan",
    "2210015": "PortScan", "2210016": "PortScan",
    "2210017": "PortScan", "2210018": "PortScan",
    "2402002": "PortScan", "2402003": "PortScan",
    
    # Botnet
    "2012648": "Bot", "2018959": "Bot", "2008581": "Bot",
    "2403388": "Bot", "2403390": "Bot", "2024364": "Bot",
    "2014411": "Bot", "2025242": "Bot",
    
    # Infiltration
    "2016538": "Infiltration", "2016540": "Infiltration",
    "2210053": "Infiltration", "2011716": "Infiltration",
    "2025757": "Infiltration",
    
    # Heartbleed
    "2019130": "Heartbleed", "2019131": "Heartbleed",
    "2210054": "Heartbleed",
}

KEYWORD_PRIORITY = [
    (["heartbleed"], "Heartbleed"),
    (["sql injection", "sql union", "sql sleep"], "Web Attack"),
    (["xss", "cross-site scripting"], "Web Attack"),
    (["ssh brute", "ssh login", "ssh password"], "SSH-Patator"),
    (["ftp brute", "ftp login", "ftp password"], "FTP-Patator"),
    (["backdoor", "webshell", "reverse shell"], "Infiltration"),
    (["ddos", "dos attack", "amplification", "flood"], "DDoS"),
    (["nmap", "portscan", "port scan"], "PortScan"),
    (["bot", "botnet", "c&c", "c2"], "Bot"),
    (["web attack", "exploit", "injection"], "Web Attack"),
]


def map_suricata_to_cicids_label(signature, category, signature_id):
    """
    Mapeia alerta Suricata para label CIC-IDS-2017.
    
    Args:
        signature: Descrição da assinatura
        category: Categoria do alerta
        signature_id: ID da assinatura
        
    Returns:
        String com label mapeado
    """
    sid_str = str(signature_id) if signature_id else None
    
    if sid_str and sid_str in SURICATA_SID_TO_CICIDS:
        return SURICATA_SID_TO_CICIDS[sid_str]
    
    sig_lower = str(signature).lower() if signature else ""
    cat_lower = str(category).lower() if category else ""
    text = f"{sig_lower} {cat_lower}"
    
    for keywords, label in KEYWORD_PRIORITY:
        if any(kw in text for kw in keywords):
            return label
    
    return "BENIGN"


# ============================================================================
# PARTE 3: EXTRAÇÃO DE FEATURES CIC-IDS-2017
# ============================================================================

def extract_cicids_features(event):
    """
    Extrai aproximação das 78 features do CIC-IDS-2017.
    
    Nota: eve.json não contém todas as estatísticas do CICFlowMeter.
    Esta função fornece aproximações baseadas em informações disponíveis.
    
    Args:
        event: Evento JSON do Suricata
        
    Returns:
        Dicionário com features extraídas
    """
    flow = event.get("flow", {})
    tcp = event.get("tcp", {})
    
    dest_port = event.get("dest_port", 0)
    flow_duration = flow.get("age", 0) * 1_000_000
    
    fwd_pkts = flow.get("pkts_toserver", 0)
    bwd_pkts = flow.get("pkts_toclient", 0)
    fwd_bytes = flow.get("bytes_toserver", 0)
    bwd_bytes = flow.get("bytes_toclient", 0)
    
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes
    
    flow_bytes_per_s = (total_bytes / (flow_duration / 1_000_000)) if flow_duration > 0 else 0
    flow_pkts_per_s = (total_pkts / (flow_duration / 1_000_000)) if flow_duration > 0 else 0
    
    fwd_pkt_len_mean = fwd_bytes / fwd_pkts if fwd_pkts > 0 else 0
    bwd_pkt_len_mean = bwd_bytes / bwd_pkts if bwd_pkts > 0 else 0
    
    fin_flag = tcp.get("fin", False)
    syn_flag = tcp.get("syn", False)
    rst_flag = tcp.get("rst", False)
    psh_flag = tcp.get("psh", False)
    ack_flag = tcp.get("ack", False)
    urg_flag = tcp.get("urg", False)
    
    features = {
        "Destination_Port": dest_port,
        "Flow_Duration": flow_duration,
        "Total_Fwd_Packets": fwd_pkts,
        "Total_Backward_Packets": bwd_pkts,
        "Total_Length_of_Fwd_Packets": fwd_bytes,
        "Total_Length_of_Bwd_Packets": bwd_bytes,
        "Fwd_Packet_Length_Max": fwd_pkt_len_mean,
        "Fwd_Packet_Length_Min": fwd_pkt_len_mean,
        "Fwd_Packet_Length_Mean": fwd_pkt_len_mean,
        "Fwd_Packet_Length_Std": 0,
        "Bwd_Packet_Length_Max": bwd_pkt_len_mean,
        "Bwd_Packet_Length_Min": bwd_pkt_len_mean,
        "Bwd_Packet_Length_Mean": bwd_pkt_len_mean,
        "Bwd_Packet_Length_Std": 0,
        "Flow_Bytes/s": flow_bytes_per_s,
        "Flow_Packets/s": flow_pkts_per_s,
        "Flow_IAT_Mean": 0,
        "Flow_IAT_Std": 0,
        "Flow_IAT_Max": 0,
        "Flow_IAT_Min": 0,
        "Fwd_IAT_Total": 0,
        "Fwd_IAT_Mean": 0,
        "Fwd_IAT_Std": 0,
        "Fwd_IAT_Max": 0,
        "Fwd_IAT_Min": 0,
        "Bwd_IAT_Total": 0,
        "Bwd_IAT_Mean": 0,
        "Bwd_IAT_Std": 0,
        "Bwd_IAT_Max": 0,
        "Bwd_IAT_Min": 0,
        "Fwd_PSH_Flags": 1 if psh_flag else 0,
        "Bwd_PSH_Flags": 0,
        "Fwd_URG_Flags": 1 if urg_flag else 0,
        "Bwd_URG_Flags": 0,
        "FIN_Flag_Count": 1 if fin_flag else 0,
        "SYN_Flag_Count": 1 if syn_flag else 0,
        "RST_Flag_Count": 1 if rst_flag else 0,
        "PSH_Flag_Count": 1 if psh_flag else 0,
        "ACK_Flag_Count": 1 if ack_flag else 0,
        "URG_Flag_Count": 1 if urg_flag else 0,
        "CWE_Flag_Count": 0,
        "ECE_Flag_Count": 0,
        "Fwd_Header_Length": 20 * fwd_pkts,
        "Bwd_Header_Length": 20 * bwd_pkts,
        "Fwd_Packets/s": (fwd_pkts / (flow_duration / 1_000_000)) if flow_duration > 0 else 0,
        "Bwd_Packets/s": (bwd_pkts / (flow_duration / 1_000_000)) if flow_duration > 0 else 0,
        "Min_Packet_Length": min(fwd_pkt_len_mean, bwd_pkt_len_mean) if fwd_pkt_len_mean > 0 and bwd_pkt_len_mean > 0 else 0,
        "Max_Packet_Length": max(fwd_pkt_len_mean, bwd_pkt_len_mean),
        "Packet_Length_Mean": (fwd_pkt_len_mean + bwd_pkt_len_mean) / 2,
        "Packet_Length_Std": 0,
        "Packet_Length_Variance": 0,
        "Down/Up_Ratio": bwd_pkts / fwd_pkts if fwd_pkts > 0 else 0,
        "Average_Packet_Size": total_bytes / total_pkts if total_pkts > 0 else 0,
        "Avg_Fwd_Segment_Size": fwd_pkt_len_mean,
        "Avg_Bwd_Segment_Size": bwd_pkt_len_mean,
        "Fwd_Avg_Bytes/Bulk": fwd_bytes,
        "Fwd_Avg_Packets/Bulk": fwd_pkts,
        "Fwd_Avg_Bulk_Rate": 0,
        "Bwd_Avg_Bytes/Bulk": bwd_bytes,
        "Bwd_Avg_Packets/Bulk": bwd_pkts,
        "Bwd_Avg_Bulk_Rate": 0,
        "Subflow_Fwd_Packets": fwd_pkts,
        "Subflow_Fwd_Bytes": fwd_bytes,
        "Subflow_Bwd_Packets": bwd_pkts,
        "Subflow_Bwd_Bytes": bwd_bytes,
        "Init_Win_bytes_forward": 0,
        "Init_Win_bytes_backward": 0,
        "act_data_pkt_fwd": fwd_pkts,
        "min_seg_size_forward": fwd_pkt_len_mean,
        "Active_Mean": 0,
        "Active_Std": 0,
        "Active_Max": 0,
        "Active_Min": 0,
        "Idle_Mean": 0,
        "Idle_Std": 0,
        "Idle_Max": 0,
        "Idle_Min": 0,
    }
    
    return features


# ============================================================================
# PARTE 4: INTEGRAÇÃO IA-IDS
# ============================================================================

def process_eve_with_ia_models(eve_path, models, scaler, feature_names, le):
    """
    Processa eve.json aplicando modelos de IA.
    
    Args:
        eve_path: Caminho do arquivo eve.json
        models: Dicionário com modelos treinados
        scaler: StandardScaler treinado
        feature_names: Lista com nomes das features
        le: LabelEncoder treinado
        
    Returns:
        Tupla (DataFrame com predições, estatísticas, SIDs não mapeados)
    """
    if not os.path.exists(eve_path):
        return pd.DataFrame(), {}, []
    
    alerts = []
    unmapped_sids = []
    stats = {
        "total_events": 0,
        "alert_events": 0,
        "features_extracted": 0,
        "features_missing": 0
    }
    
    with open(eve_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stats["total_events"] += 1
            
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if event.get("event_type") != "alert":
                continue
            
            stats["alert_events"] += 1
            
            alert = event.get("alert", {})
            sig_id = alert.get("signature_id")
            sig = alert.get("signature")
            cat = alert.get("category")
            
            features = extract_cicids_features(event)
            stats["features_extracted"] += 1
            
            suricata_label = map_suricata_to_cicids_label(sig, cat, sig_id)
            
            if suricata_label == "BENIGN" and sig_id and sig:
                sid_str = str(sig_id)
                if sid_str not in SURICATA_SID_TO_CICIDS:
                    unmapped_sids.append({
                        'SID': sig_id,
                        'Signature': sig,
                        'Category': cat
                    })
            
            alerts.append({
                "timestamp": event.get("timestamp"),
                "src_ip": event.get("src_ip"),
                "dest_ip": event.get("dest_ip"),
                "suricata_signature": sig,
                "suricata_category": cat,
                "suricata_signature_id": sig_id,
                "suricata_label": suricata_label,
                **features
            })
    
    if not alerts:
        return pd.DataFrame(), stats, unmapped_sids
    
    df = pd.DataFrame(alerts)
    
    X = pd.DataFrame(0, index=df.index, columns=feature_names)
    
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            stats["features_missing"] += 1
    
    X_scaled = scaler.transform(X)
    
    for model_name, model in models.items():
        y_pred = model.predict(X_scaled)
        df[f"ia_pred_{model_name}_encoded"] = y_pred
        df[f"ia_pred_{model_name}_label"] = le.inverse_transform(y_pred)
    
    return df, stats, unmapped_sids


# ============================================================================
# PARTE 5: CORRELAÇÃO TEMPORAL E ANÁLISE COMPARATIVA
# ============================================================================

def parse_timestamp(ts_str):
    """Parse timestamp robusto."""
    if pd.isna(ts_str):
        return None
    
    try:
        ts_str = str(ts_str).strip()
        
        for fmt in [
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
        ]:
            try:
                return pd.to_datetime(ts_str, format=fmt)
            except:
                continue
        
        return pd.to_datetime(ts_str)
    except:
        return None


def calculate_suricata_metrics_temporal(runs_dir, ground_truth_path, temporal_window=5):
    """
    Calcula TPR/FPR do Suricata usando correlação temporal otimizada.
    
    Args:
        runs_dir: Diretório com runs do Suricata
        ground_truth_path: Caminho do CSV com ground truth
        temporal_window: Janela temporal em segundos
        
    Returns:
        DataFrame com métricas por PCAP
    """
    logging.info("Calculando métricas do Suricata com correlação temporal")
    logging.info(f"Janela temporal: ±{temporal_window} segundos")
    
    if not os.path.exists(runs_dir):
        logging.warning(f"Diretório não encontrado: {runs_dir}")
        return pd.DataFrame()
    
    try:
        gt = pd.read_csv(ground_truth_path, low_memory=False)
        
        if 'Timestamp' not in gt.columns and ' Timestamp' not in gt.columns:
            logging.warning("Coluna Timestamp ausente - usando aproximação")
            return pd.DataFrame()
        
        if ' Timestamp' in gt.columns:
            gt.rename(columns={' Timestamp': 'Timestamp'}, inplace=True)
        
        gt['Timestamp_parsed'] = pd.to_datetime(gt['Timestamp'], errors='coerce')
        gt = gt[gt['Timestamp_parsed'].notna()]
        
        if len(gt) == 0:
            logging.warning("Nenhum timestamp válido no ground truth")
            return pd.DataFrame()
        
        gt = gt.sort_values('Timestamp_parsed').reset_index(drop=True)
        
        logging.info(f"Ground truth: {len(gt):,} registros")
        
        gt_attacks = gt[gt['Label'] != 'BENIGN'].copy()
        gt_benign = gt[gt['Label'] == 'BENIGN'].copy()
        
        logging.info(f"Ataques: {len(gt_attacks):,}")
        logging.info(f"Benignos: {len(gt_benign):,}")
        
    except Exception as e:
        logging.error(f"Erro ao carregar ground truth: {e}")
        return pd.DataFrame()
    
    results = []
    window_td = pd.Timedelta(seconds=temporal_window)
    
    for pcap_name in sorted(os.listdir(runs_dir)):
        pcap_dir = os.path.join(runs_dir, pcap_name)
        
        if not os.path.isdir(pcap_dir):
            continue
        
        logging.info(f"Processando {pcap_name}...")
        
        eve_path = os.path.join(pcap_dir, "eve.json")
        
        if not os.path.exists(eve_path):
            logging.warning(f"eve.json ausente em {pcap_name}")
            continue
        
        suricata_alerts = []
        
        with open(eve_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                    if event.get("event_type") == "alert":
                        ts = parse_timestamp(event.get("timestamp"))
                        if ts:
                            suricata_alerts.append({
                                'timestamp': ts,
                                'signature': event.get('alert', {}).get('signature', '')
                            })
                except:
                    continue
        
        if not suricata_alerts:
            logging.info(f"Nenhum alerta com timestamp em {pcap_name}")
            continue
        
        suricata_df = pd.DataFrame(suricata_alerts)
        suricata_df = suricata_df.sort_values('timestamp').reset_index(drop=True)
        
        logging.info(f"Alertas do Suricata: {len(suricata_df)}")
        
        merged = pd.merge_asof(
            suricata_df.sort_values('timestamp'),
            gt_attacks[['Timestamp_parsed', 'Label']].sort_values('Timestamp_parsed'),
            left_on='timestamp',
            right_on='Timestamp_parsed',
            direction='nearest',
            tolerance=window_td
        )
        
        TP = merged['Label'].notna().sum()
        FP = merged['Label'].isna().sum()
        
        undetected = pd.merge_asof(
            gt_attacks[['Timestamp_parsed', 'Label']].sort_values('Timestamp_parsed'),
            suricata_df[['timestamp']].sort_values('timestamp'),
            left_on='Timestamp_parsed',
            right_on='timestamp',
            direction='nearest',
            tolerance=window_td
        )
        
        FN = undetected['timestamp'].isna().sum()
        TN = len(gt_benign)
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        results.append({
            "pcap": pcap_name,
            "suricata_alerts": len(suricata_df),
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "TN": int(TN),
            "TPR": float(tpr),
            "FPR": float(fpr),
            "Precision": float(precision)
        })
        
        logging.info(f"TP={TP}, FP={FP}, FN={FN}")
        logging.info(f"TPR={tpr*100:.2f}%, FPR={fpr*100:.4f}%")
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        total_tp = df['TP'].sum()
        total_fp = df['FP'].sum()
        total_fn = df['FN'].sum()
        total_tn = df['TN'].sum()
        
        overall_tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        
        logging.info(f"\nMétricas agregadas:")
        logging.info(f"TPR: {overall_tpr*100:.2f}%")
        logging.info(f"FPR: {overall_fpr*100:.4f}%")
        logging.info(f"Precision: {overall_precision*100:.2f}%")
        
        df['Overall_TPR'] = overall_tpr
        df['Overall_FPR'] = overall_fpr
        df['Overall_Precision'] = overall_precision
    
    return df


def create_comparison_table(ia_metrics, suricata_metrics):
    """
    Cria tabela comparativa IA vs IDS.
    
    Args:
        ia_metrics: DataFrame com métricas de IA
        suricata_metrics: DataFrame com métricas do Suricata
        
    Returns:
        DataFrame com comparação
    """
    rows = []
    
    if not ia_metrics.empty:
        for _, r in ia_metrics.iterrows():
            rows.append({
                "Fonte": "IA",
                "Modelo": r.get("Model", ""),
                "Accuracy": round(float(r.get("Accuracy", 0)) * 100, 2),
                "TPR": round(float(r.get("Macro_Recall(TPR)", 0)) * 100, 2),
                "FPR": round(float(r.get("Macro_FPR", 0)) * 100, 2),
                "F1": round(float(r.get("Macro_F1", 0)) * 100, 2)
            })
    
    if not suricata_metrics.empty and 'Overall_TPR' in suricata_metrics.columns:
        overall_tpr = suricata_metrics['Overall_TPR'].iloc[0]
        overall_fpr = suricata_metrics['Overall_FPR'].iloc[0]
        
        rows.append({
            "Fonte": "IDS",
            "Modelo": "Suricata",
            "Accuracy": "",
            "TPR": round(overall_tpr * 100, 2),
            "FPR": round(overall_fpr * 100, 4),
            "F1": ""
        })
    
    return pd.DataFrame(rows)


def plot_ia_vs_ids_comparison(comp_df, output_path):
    """Plota comparação de métricas IA vs IDS."""
    if comp_df.empty:
        return
    
    plot_df = comp_df[comp_df["TPR"].notna() & (comp_df["TPR"] != "")].copy()
    
    if plot_df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = plot_df["Fonte"] + " - " + plot_df["Modelo"]
    tpr = plot_df["TPR"].astype(float)
    
    colors = ['steelblue', 'coral', 'seagreen', 'purple']
    bars1 = ax1.bar(range(len(models)), tpr, color=colors[:len(models)], alpha=0.8)
    
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel("TPR (%)")
    ax1.set_title("Taxa de Detecao (TPR) - IA vs IDS")
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    fpr = plot_df["FPR"].astype(float)
    
    bars2 = ax2.bar(range(len(models)), fpr, color=colors[:len(models)], alpha=0.8)
    
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel("FPR (%)")
    ax2.set_title("Taxa de Falsos Positivos (FPR) - IA vs IDS")
    ax2.set_ylim([0, max(fpr.max() * 1.2, 5)])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    
    logging.info(f"Gráfico comparativo salvo em: {output_path}")


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal de execução."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    logging.info("="*70)
    logging.info("SCRIPT 03: INTEGRACAO E ANALISE IDS")
    logging.info("="*70)
    
    args = parse_args(config)
    
    models_dir = config["paths"]["models_dir"]
    results_dir = config["paths"]["results_dir"]
    runs_dir = args.runs_dir
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    
    # PARTE 1: Processamento com Suricata
    if not args.skip_suricata:
        logging.info("\n" + "="*70)
        logging.info("PROCESSAMENTO DE PCAPS COM SURICATA")
        logging.info("="*70)
        
        update_suricata_rules(config["suricata"]["skip_update"])
        
        pcap_files = sorted(
            glob.glob(os.path.join(args.pcap_dir, "*.pcap")) +
            glob.glob(os.path.join(args.pcap_dir, "*.pcapng"))
        )
        
        if not pcap_files:
            logging.error(f"Nenhum PCAP encontrado em {args.pcap_dir}")
            return 1
        
        logging.info(f"Encontrados {len(pcap_files)} arquivos PCAP")
        
        csv_paths = []
        
        for pcap_path in pcap_files:
            base_name = os.path.splitext(os.path.basename(pcap_path))[0]
            output_dir = os.path.join(runs_dir, base_name)
            
            eve_path = process_pcap_with_suricata(
                pcap_path,
                output_dir,
                config["suricata"]["binary"]
            )
            
            df_alerts = parse_eve_to_dataframe(eve_path)
            
            csv_path = os.path.join(output_dir, f"{base_name}_alerts.csv")
            df_alerts.to_csv(csv_path, index=False)
            csv_paths.append(csv_path)
            
            logging.info(f"Extraídos {len(df_alerts)} alertas de {base_name}")
        
        summary_path = os.path.join(results_dir, "suricata_summary_by_category.csv")
        aggregate_alerts_by_category(csv_paths, summary_path)
    
    else:
        logging.info("Processamento de PCAPs pulado (--skip-suricata)")
    
    # PARTE 2: Integração IA-IDS
    logging.info("\n" + "="*70)
    logging.info("INTEGRACAO IA-IDS")
    logging.info("="*70)
    
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
    
    with open(os.path.join(models_dir, "features_list.json"), "r") as f:
        feature_names = json.load(f)
    
    models = {}
    for model_name in ["DecisionTree", "RandomForest", "MLP"]:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
            logging.info(f"Modelo carregado: {model_name}")
    
    if not models:
        logging.error("Nenhum modelo encontrado. Execute 02_train_evaluate.py primeiro")
        return 1
    
    eve_files = []
    for root, dirs, files in os.walk(runs_dir):
        if "eve.json" in files:
            eve_files.append(os.path.join(root, "eve.json"))
    
    if not eve_files:
        logging.error(f"Nenhum eve.json encontrado em {runs_dir}")
        return 1
    
    logging.info(f"Encontrados {len(eve_files)} arquivos eve.json")
    
    all_predictions = {}
    all_extraction_stats = {}
    all_unmapped_sids = []
    
    for eve_path in eve_files:
        pcap_name = os.path.basename(os.path.dirname(eve_path))
        logging.info(f"\nProcessando {pcap_name}...")
        
        df, stats, unmapped_sids = process_eve_with_ia_models(
            eve_path, models, scaler, feature_names, le
        )
        
        if not df.empty:
            all_predictions[pcap_name] = df
            all_extraction_stats[pcap_name] = stats
            all_unmapped_sids.extend(unmapped_sids)
            
            logging.info(f"Processados {len(df)} alertas")
    
    for model_name in models.keys():
        model_predictions = []
        
        for pcap_name, df in all_predictions.items():
            df_model = df[["timestamp", "src_ip", "dest_ip", 
                          "suricata_label", f"ia_pred_{model_name}_label"]].copy()
            df_model["pcap_name"] = pcap_name
            model_predictions.append(df_model)
        
        if model_predictions:
            combined = pd.concat(model_predictions, ignore_index=True)
            output_path = os.path.join(results_dir, f"ia_ids_integration_{model_name}.csv")
            combined.to_csv(output_path, index=False)
            logging.info(f"Predições de {model_name} salvas em: {output_path}")
    
    quality_rows = []
    for pcap_name, stats in all_extraction_stats.items():
        quality_rows.append({
            "pcap": pcap_name,
            "total_events": stats.get("total_events", 0),
            "alert_events": stats.get("alert_events", 0),
            "features_extracted": stats.get("features_extracted", 0),
            "features_missing": stats.get("features_missing", 0),
            "extraction_rate_pct": (stats.get("features_extracted", 0) / 78 * 100) 
                                   if stats.get("features_extracted") else 0
        })
    
    if quality_rows:
        quality_df = pd.DataFrame(quality_rows)
        quality_path = os.path.join(results_dir, "feature_extraction_quality.csv")
        quality_df.to_csv(quality_path, index=False)
        logging.info(f"Qualidade da extração salva em: {quality_path}")
    
    if all_unmapped_sids:
        sid_counts = Counter(item['SID'] for item in all_unmapped_sids)
        
        unique_sids = {}
        for item in all_unmapped_sids:
            sid = item['SID']
            if sid not in unique_sids:
                unique_sids[sid] = {
                    'SID': sid,
                    'Signature': item['Signature'],
                    'Category': item['Category'],
                    'Frequency': sid_counts[sid]
                }
        
        unmapped_df = pd.DataFrame(list(unique_sids.values()))
        unmapped_df = unmapped_df.sort_values('Frequency', ascending=False)
        
        unmapped_path = os.path.join(results_dir, "unmapped_sids_summary.csv")
        unmapped_df.to_csv(unmapped_path, index=False)
        logging.info(f"SIDs não mapeados salvos em: {unmapped_path}")
    
    # PARTE 3: Análise Comparativa
    logging.info("\n" + "="*70)
    logging.info("ANALISE COMPARATIVA IA VS IDS")
    logging.info("="*70)
    
    suricata_metrics = calculate_suricata_metrics_temporal(
        runs_dir,
        config["paths"]["combined_csv"],
        args.temporal_window
    )
    
    if not suricata_metrics.empty:
        suricata_path = os.path.join(results_dir, "suricata_vs_groundtruth.csv")
        suricata_metrics.to_csv(suricata_path, index=False)
        logging.info(f"Métricas do Suricata salvas em: {suricata_path}")
    
    ia_metrics_path = os.path.join(results_dir, "overall_metrics.csv")
    if os.path.exists(ia_metrics_path):
        ia_metrics = pd.read_csv(ia_metrics_path)
    else:
        ia_metrics = pd.DataFrame()
    
    comp_df = create_comparison_table(ia_metrics, suricata_metrics)
    
    if not comp_df.empty:
        comp_path = os.path.join(results_dir, "ia_vs_ids_comparison.csv")
        comp_df.to_csv(comp_path, index=False)
        logging.info(f"Comparação IA vs IDS salva em: {comp_path}")
        
        with open(os.path.join(results_dir, "ia_vs_ids_comparison.tex"), "w") as f:
            f.write(comp_df.to_latex(
                index=False,
                caption="Comparacao IA vs IDS (percentual)",
                label="tab:ia_vs_ids",
                escape=False
            ))
        
        plot_path = os.path.join(results_dir, "ia_vs_ids_metrics.png")
        plot_ia_vs_ids_comparison(comp_df, plot_path)
    
    logging.info("\n" + "="*70)
    logging.info("INTEGRACAO E ANALISE IDS CONCLUIDA")
    logging.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())