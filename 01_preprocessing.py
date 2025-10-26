#!/usr/bin/env python3
"""
Script 01: Preprocessamento e Consolidação do Dataset

Descrição:
    Combina múltiplos arquivos CSV do dataset CIC-IDS-2017 em um arquivo único,
    realizando normalização de dados, tratamento de valores ausentes e
    imputação estatística.

Funcionalidades:
    - Normalização de cabeçalhos de colunas
    - Garantia de coluna "Label" em todos os arquivos
    - Tratamento de valores NaN e infinitos
    - Imputação por mediana para dados numéricos
    - Remoção de linhas com dados insuficientes
    - Geração de estatísticas descritivas

Uso:
    python3 01_preprocessing.py
    python3 01_preprocessing.py --csv-dir path/to/CSVs --out custom.csv

Autor: Bernardo Vivian Vieira
Data: Outubro 2025
"""

import argparse
import glob
import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def setup_logging(level="INFO"):
    """
    Configura sistema de logging.
    
    Args:
        level (str): Nível de log (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s] %(message)s"
    )


def load_config():
    """
    Carrega arquivo de configuração YAML.
    
    Returns:
        dict: Dicionário com configurações do projeto
        
    Raises:
        SystemExit: Se config.yaml não encontrado
    """
    if not os.path.exists("config.yaml"):
        logging.error("config.yaml não encontrado. Execute 00_sanity_check.py primeiro")
        sys.exit(1)
    
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def parse_args(config):
    """
    Processa argumentos da linha de comando.
    
    Args:
        config (dict): Configurações carregadas
        
    Returns:
        argparse.Namespace: Argumentos processados
    """
    parser = argparse.ArgumentParser(
        description="Consolidação e normalização de arquivos CSV do CIC-IDS-2017"
    )
    
    parser.add_argument(
        "--csv-dir",
        default=config["paths"]["csv_dir"],
        help="Diretório contendo arquivos CSV"
    )
    
    parser.add_argument(
        "--out",
        default=config["paths"]["combined_csv"],
        help="Arquivo CSV combinado de saída"
    )
    
    parser.add_argument(
        "--min-non-na",
        type=float,
        default=config["preprocessing"]["min_non_na"],
        help="Percentual mínimo de colunas não-NaN para manter linha (0.0-1.0)"
    )
    
    return parser.parse_args()


def normalize_columns(df):
    """
    Remove espaços extras e garante nomes únicos de colunas.
    
    Args:
        df (pd.DataFrame): DataFrame a ser normalizado
        
    Returns:
        pd.DataFrame: DataFrame com colunas normalizadas
    """
    cols = [c.strip().replace("..", ".").replace(".1", "") for c in df.columns]
    
    seen = {}
    new_cols = []
    for c in cols:
        base = c
        i = 1
        while c in seen:
            c = f"{base}_{i}"
            i += 1
        seen[c] = True
        new_cols.append(c)
    
    df.columns = new_cols
    return df


def detect_label_column(df, filename):
    """
    Detecta e normaliza coluna de label.
    
    Args:
        df (pd.DataFrame): DataFrame a ser processado
        filename (str): Nome do arquivo (para logging)
        
    Returns:
        pd.DataFrame: DataFrame com coluna "Label", ou None se não encontrada
    """
    label_cols = [c for c in df.columns if c.strip().lower() == "label"]
    
    if not label_cols:
        alt = [c for c in df.columns if "label" in c.lower()]
        if alt:
            logging.warning(
                f"Coluna label encontrada como '{alt[0]}' em {filename}; "
                f"renomeando para 'Label'"
            )
            df.rename(columns={alt[0]: "Label"}, inplace=True)
        else:
            logging.warning(f"Nenhuma coluna 'Label' em {filename} - arquivo ignorado")
            return None
    else:
        df.rename(columns={label_cols[0]: "Label"}, inplace=True)
    
    return df


def safe_read_csv(path):
    """
    Lê arquivo CSV de forma segura, tratando erros.
    
    Args:
        path (str): Caminho do arquivo CSV
        
    Returns:
        pd.DataFrame: DataFrame carregado, ou None em caso de erro
    """
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        logging.error(f"Falha ao ler {path}: {e}")
        return None


def clean_dataframe(df, filename, min_non_na):
    """
    Limpa DataFrame: trata valores ausentes e infinitos.
    
    Args:
        df (pd.DataFrame): DataFrame a ser limpo
        filename (str): Nome do arquivo (para logging)
        min_non_na (float): Percentual mínimo de dados não-NaN
        
    Returns:
        pd.DataFrame: DataFrame limpo
    """
    df["Label"] = df["Label"].astype(str).str.strip()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    thresh = int(df.shape[1] * min_non_na) if df.shape[1] > 0 else 0
    before = len(df)
    df = df.dropna(thresh=thresh)
    after = len(df)
    
    if before > after:
        logging.info(
            f"{filename}: removidas {before-after} linhas por dados insuficientes "
            f"(<{min_non_na*100:.0f}% preenchimento)"
        )
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(medians)
    
    return df


def main():
    """
    Função principal de preprocessamento.
    
    Returns:
        int: 0 se bem-sucedido, 1 caso contrário
    """
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    args = parse_args(config)
    csv_dir = args.csv_dir
    out = args.out
    min_non_na = args.min_non_na
    
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    logging.info(f"Encontrados {len(files)} arquivos CSV em: {csv_dir}")
    
    if not files:
        logging.error("Nenhum arquivo CSV encontrado. Verifique o caminho --csv-dir")
        return 1
    
    dfs = []
    for f in files:
        filename = os.path.basename(f)
        logging.info(f"Processando {filename}...")
        
        df = safe_read_csv(f)
        if df is None:
            continue
        
        df = normalize_columns(df)
        
        df = detect_label_column(df, filename)
        if df is None:
            continue
        
        df = clean_dataframe(df, filename, min_non_na)
        
        dfs.append(df)
    
    if not dfs:
        logging.error("Nenhum dataframe válido após processamento")
        return 1
    
    logging.info("Combinando DataFrames...")
    combined = pd.concat(dfs, ignore_index=True)
    
    cols = [c for c in combined.columns if c != "Label"] + ["Label"]
    combined = combined[cols]
    
    combined.to_csv(out, index=False)
    
    logging.info(f"Arquivo combinado salvo em: {out}")
    logging.info(f"Total de linhas: {len(combined):,}")
    logging.info(f"Total de colunas: {len(combined.columns)}")
    
    logging.info("\nDistribuição de classes (top 20):")
    class_counts = combined["Label"].value_counts().head(20)
    for label, count in class_counts.items():
        percentage = (count / len(combined)) * 100
        logging.info(f"  {label}: {count:,} ({percentage:.2f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())