#!/usr/bin/env python3
"""
Script 02: Treinamento, Avaliação e Validação Cruzada

Descrição:
    Script unificado que realiza treinamento de modelos de machine learning,
    exportação de métricas de desempenho e validação cruzada estratificada.

Funcionalidades:
    - Preprocessamento de dados (remoção de não-numéricos, normalização)
    - Treinamento de três modelos de classificação
    - Cálculo de métricas: Accuracy, Precision, Recall, F1, TPR, FPR
    - Validação cruzada estratificada
    - Exportação de artefatos (modelos, scalers, encoders)
    - Geração de visualizações (matrizes de confusão)

Saída:
    models/*.pkl, results/*.csv, results/*.png, results/*.tex

Uso:
    python3 02_train_evaluate.py
    python3 02_train_evaluate.py --skip-cv
    python3 02_train_evaluate.py --test-size 0.2

Autor: Bernardo Vivian Vieira
Data: Outubro 2025
"""

import argparse
import json
import os
import sys
import logging
import joblib
import time
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        logging.error("config.yaml não encontrado")
        sys.exit(1)
    
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def parse_args(config):
    """Processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treinamento e avaliação de modelos de machine learning"
    )
    
    parser.add_argument(
        "--data",
        default=config["paths"]["combined_csv"],
        help="Arquivo CSV combinado (com coluna Label)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=config["test_size"],
        help="Proporção do conjunto de teste"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=config["cv_folds"],
        help="Número de folds para validação cruzada"
    )
    
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Pula validação cruzada"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=config["random_seed"],
        help="Semente aleatória"
    )
    
    return parser.parse_args()


def load_and_preprocess(data_path, models_dir):
    """Carrega e preprocessa dados do arquivo CSV."""
    logging.info(f"Carregando dados de {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    df = pd.read_csv(data_path, low_memory=False)
    logging.info(f"  Linhas: {len(df):,} | Colunas: {len(df.columns)}")
    
    if "Label" not in df.columns:
        raise RuntimeError("Coluna 'Label' ausente")
    
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "Label"]
    if obj_cols:
        logging.info(f"  Removendo {len(obj_cols)} colunas não-numéricas")
        df = df.drop(columns=obj_cols, errors='ignore')
    
    df = df.fillna(df.median(numeric_only=True))
    
    X = df.drop(columns=["Label"])
    y = df["Label"].astype(str)
    
    logging.info(f"  Features: {X.shape[1]} | Amostras: {X.shape[0]:,}")
    logging.info(f"  Classes: {y.nunique()}")
    
    os.makedirs(models_dir, exist_ok=True)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    with open(os.path.join(models_dir, "features_list.json"), "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)
    
    logging.info("  Preprocessamento concluído")
    
    return X_scaled, y_enc, X.columns.tolist(), le


def create_models(config):
    """Cria instâncias dos modelos."""
    models = {}
    
    models["DecisionTree"] = DecisionTreeClassifier(
        **config["models"]["DecisionTree"]
    )
    
    models["RandomForest"] = RandomForestClassifier(
        **config["models"]["RandomForest"]
    )
    
    mlp_config = config["models"]["MLP"].copy()
    mlp_config["hidden_layer_sizes"] = tuple(mlp_config["hidden_layer_sizes"])
    models["MLP"] = MLPClassifier(**mlp_config)
    
    return models


def plot_confusion_matrix(cm, model_name, out_dir):
    """Plota e salva matriz de confusão."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"{model_name} - Matriz de Confusão")
    plt.colorbar()
    plt.xlabel("Predição")
    plt.ylabel("Real")
    
    if cm.shape[0] <= 20:
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center',
                    color='white' if val > cm.max()/2 else 'black',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_confusion.png"), dpi=150)
    plt.close()


def calculate_tpr_fpr_macro(cm):
    """Calcula TPR e FPR macro."""
    num_classes = cm.shape[0]
    tprs, fprs = [], []
    
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    return float(np.mean(tprs)), float(np.mean(fprs))


def train_and_evaluate(models, X_train, X_test, y_train, y_test, 
                       models_dir, results_dir, class_names):
    """Treina e avalia modelos."""
    logging.info("\n" + "="*70)
    logging.info("TREINAMENTO E AVALIACAO")
    logging.info("="*70)
    
    os.makedirs(results_dir, exist_ok=True)
    
    overall_rows = []
    
    for name, model in models.items():
        logging.info(f"\nTreinando {name}...")
        
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        
        t0 = time.time()
        y_pred = model.predict(X_test)
        t_infer = time.time() - t0
        t_infer_per_sample = t_infer / len(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True, 
            zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        tpr_macro, fpr_macro = calculate_tpr_fpr_macro(cm)
        
        logging.info(f"  Tempo treino: {t_train:.2f}s")
        logging.info(f"  Tempo inferência: {t_infer_per_sample*1000:.4f}ms/amostra")
        logging.info(f"  Acurácia: {acc:.4f}")
        logging.info(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
        
        joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))
        
        with open(os.path.join(results_dir, f"{name}_report.txt"), "w") as f:
            f.write(classification_report(
                y_test, y_pred, 
                target_names=class_names,
                zero_division=0
            ))
        
        plot_confusion_matrix(cm, name, results_dir)
        
        perclass_rows = []
        for cls in class_names:
            cls_info = report.get(cls, {
                "precision": 0, "recall": 0, "f1-score": 0, "support": 0
            })
            
            perclass_rows.append({
                "Model": name,
                "Class": cls,
                "Precision": float(cls_info.get("precision", 0)),
                "Recall(TPR)": float(cls_info.get("recall", 0)),
                "F1": float(cls_info.get("f1-score", 0)),
                "Support": int(cls_info.get("support", 0))
            })
        
        pd.DataFrame(perclass_rows).to_csv(
            os.path.join(results_dir, f"{name}_perclass.csv"),
            index=False
        )
        
        overall_rows.append({
            "Model": name,
            "Accuracy": acc,
            "Macro_Precision": report['macro avg']['precision'],
            "Macro_Recall(TPR)": tpr_macro,
            "Macro_F1": report['macro avg']['f1-score'],
            "Macro_FPR": fpr_macro,
            "Train_Time_s": t_train,
            "Infer_Time_ms": t_infer_per_sample * 1000
        })
    
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(os.path.join(results_dir, "overall_metrics.csv"), index=False)
    
    latex_df = overall_df[["Model", "Accuracy", "Macro_Precision", 
                          "Macro_Recall(TPR)", "Macro_F1", "Macro_FPR"]].copy()
    for col in latex_df.columns[1:]:
        latex_df[col] = (latex_df[col] * 100).round(2)
    
    with open(os.path.join(results_dir, "overall_table.tex"), "w") as f:
        f.write(latex_df.to_latex(
            index=False,
            caption="Métricas gerais por modelo (percentual)",
            label="tab:overall",
            escape=False
        ))
    
    return overall_df


def cross_validation(X, y, models, k, random_state, results_dir):
    """Executa validação cruzada k-fold."""
    logging.info("\n" + "="*70)
    logging.info(f"VALIDACAO CRUZADA ({k}-FOLD)")
    logging.info("="*70)
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    all_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"  Fold {fold_idx}/{k}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        for model_name, model in models.items():
            clf = type(model)(**model.get_params())
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            report = classification_report(
                y_test, y_pred, 
                output_dict=True,
                zero_division=0
            )
            
            all_results.append({
                "Fold": fold_idx,
                "Model": model_name,
                "Accuracy": report["accuracy"],
                "Macro_Precision": report["macro avg"]["precision"],
                "Macro_Recall(TPR)": report["macro avg"]["recall"],
                "Macro_F1": report["macro avg"]["f1-score"]
            })
    
    df_folds = pd.DataFrame(all_results)
    df_folds.to_csv(os.path.join(results_dir, "cv_folds_detail.csv"), index=False)
    
    agg = df_folds.groupby("Model")[
        ["Accuracy", "Macro_Precision", "Macro_Recall(TPR)", "Macro_F1"]
    ].agg(['mean', 'std'])
    
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg_df = agg.reset_index()
    
    agg_df.to_csv(os.path.join(results_dir, "cv_overall_metrics.csv"), index=False)
    
    latex_rows = []
    for _, row in agg_df.iterrows():
        formatted = {"Model": row["Model"]}
        
        for metric in ["Accuracy", "Macro_Precision", "Macro_Recall(TPR)", "Macro_F1"]:
            mean = row[f"{metric}_mean"] * 100
            std = row[f"{metric}_std"] * 100
            formatted[metric] = f"{mean:.2f} $\\pm$ {std:.2f}"
        
        latex_rows.append(formatted)
    
    latex_df = pd.DataFrame(latex_rows)
    
    with open(os.path.join(results_dir, "cv_overall_table.tex"), "w") as f:
        f.write(latex_df.to_latex(
            index=False,
            caption="Validação cruzada (percentual)",
            label="tab:cv",
            escape=False
        ))
    
    logging.info("\nResumo da validação cruzada:")
    for _, row in agg_df.iterrows():
        logging.info(f"{row['Model']}: Accuracy {row['Accuracy_mean']:.4f} +/- {row['Accuracy_std']:.4f}")
    
    return agg_df


def main():
    """Função principal."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    logging.info("="*70)
    logging.info("TREINAMENTO E AVALIACAO UNIFICADO")
    logging.info("="*70)
    
    args = parse_args(config)
    
    models_dir = config["paths"]["models_dir"]
    results_dir = config["paths"]["results_dir"]
    
    X, y, feature_names, le = load_and_preprocess(args.data, models_dir)
    class_names = list(le.classes_)
    
    logging.info("\nDivisão treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )
    
    logging.info(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}")
    
    os.makedirs(results_dir, exist_ok=True)
    joblib.dump(
        (X_train, X_test, y_train, y_test),
        os.path.join(results_dir, "train_test_splits.pkl")
    )
    
    models = create_models(config)
    
    overall_df = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        models_dir, results_dir, class_names
    )
    
    if not args.skip_cv:
        cv_df = cross_validation(
            X, y, models,
            args.cv_folds, args.random_state, results_dir
        )
    else:
        logging.info("\nValidação cruzada pulada (--skip-cv)")
    
    logging.info("\n" + "="*70)
    logging.info("PIPELINE CONCLUIDO")
    logging.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())