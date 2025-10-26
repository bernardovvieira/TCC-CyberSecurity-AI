#!/usr/bin/env python3
"""
Script 04: Ataques Adversariais e Análise Qualitativa

Descrição:
    Script unificado que implementa ataques adversariais (FGSM, PGD, C&W L2)
    e realiza análise qualitativa completa dos resultados, incluindo
    vulnerabilidade por feature, padrões de misclassificação e transferibilidade.

Funcionalidades:
    - Treinamento de modelo surrogate (MLP)
    - Implementação de ataques adversariais:
        * FGSM (Fast Gradient Sign Method)
        * PGD (Projected Gradient Descent)
        * C&W L2 (Carlini & Wagner L2)
    - Avaliação de transferibilidade entre modelos
    - Análise de perturbações por feature
    - Identificação de classes mais vulneráveis
    - Análise de padrões de misclassificação

Saída:
    - results/adv_by_epsilon.csv
    - results/adv_overall_metrics.csv
    - results/adv_comparison.png
    - results/adv_by_attack_type.png
    - results/adv_perturbation_analysis.csv
    - results/adv_feature_vulnerability.png
    - results/adv_misclassification_patterns.txt

Requisitos:
    - PyTorch (torch, torchvision)

Uso:
    python3 04_adversarial.py
    python3 04_adversarial.py --eps-list 0.01,0.05,0.1
    python3 04_adversarial.py --attack-types fgsm,pgd,cw_l2

Autor: Bernardo Vivian Vieira
Data: Outubro 2025
"""

import argparse
import logging
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report


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
    adv_config = config["adversarial"]
    
    parser = argparse.ArgumentParser(
        description="Avaliação de ataques adversariais em modelos de IA"
    )
    
    parser.add_argument(
        "--eps-list",
        type=str,
        default=",".join(map(str, adv_config["eps_values"])),
        help="Lista de epsilons separados por vírgula"
    )
    
    parser.add_argument(
        "--attack-types",
        type=str,
        default=",".join(adv_config["attack_types"]),
        help="Tipos de ataque separados por vírgula (fgsm,pgd,cw_l2)"
    )
    
    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=adv_config["pgd_steps"],
        help="Número de iterações do PGD"
    )
    
    parser.add_argument(
        "--pgd-alpha",
        type=float,
        default=adv_config["pgd_alpha"],
        help="Tamanho do passo do PGD"
    )
    
    parser.add_argument(
        "--cw-steps",
        type=int,
        default=adv_config.get("cw_steps", 100),
        help="Número de iterações do C&W"
    )
    
    parser.add_argument(
        "--cw-lr",
        type=float,
        default=adv_config.get("cw_lr", 0.01),
        help="Learning rate do C&W"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=adv_config["surrogate_epochs"],
        help="Épocas para treinar modelo surrogate"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=adv_config["surrogate_lr"],
        help="Learning rate do surrogate"
    )
    
    return parser.parse_args()


# ============================================================================
# PARTE 1: MODELO SURROGATE
# ============================================================================

class SurrogateMLP(nn.Module):
    """
    Rede neural surrogate para geração de exemplos adversariais.
    
    Arquitetura:
        - Camada de entrada: tamanho variável
        - Camada oculta 1: 128 neurônios + ReLU + Dropout(0.2)
        - Camada oculta 2: 64 neurônios + ReLU + Dropout(0.2)
        - Camada de saída: número de classes
    """
    
    def __init__(self, in_dim, n_classes):
        """
        Inicializa modelo surrogate.
        
        Args:
            in_dim: Dimensão de entrada (número de features)
            n_classes: Número de classes
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.net(x)


def train_surrogate(model, X_train, y_train, epochs=10, lr=1e-3, batch_size=1024):
    """
    Treina modelo surrogate.
    
    Args:
        model: Instância do SurrogateMLP
        X_train: Features de treinamento
        y_train: Labels de treinamento
        epochs: Número de épocas
        lr: Learning rate
        batch_size: Tamanho do batch
    """
    logging.info("Treinando modelo surrogate")
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        logging.info(f"Epoca {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
    
    logging.info("Surrogate treinado com sucesso")


# ============================================================================
# PARTE 2: ATAQUES ADVERSARIAIS
# ============================================================================

def fgsm_attack(model, x, y, eps):
    """
    Fast Gradient Sign Method.
    
    Args:
        model: Modelo alvo
        x: Tensor com dados de entrada
        y: Tensor com labels
        eps: Magnitude da perturbação
        
    Returns:
        Tensor com exemplos adversariais
    """
    x = x.clone().detach().requires_grad_(True)
    
    outputs = model(x)
    loss = nn.CrossEntropyLoss()(outputs, y)
    loss.backward()
    
    perturbation = eps * x.grad.sign()
    adv_x = x + perturbation
    
    return adv_x.detach()


def pgd_attack(model, x, y, eps, alpha, steps):
    """
    Projected Gradient Descent.
    
    Args:
        model: Modelo alvo
        x: Tensor com dados de entrada
        y: Tensor com labels
        eps: Magnitude máxima da perturbação
        alpha: Tamanho do passo
        steps: Número de iterações
        
    Returns:
        Tensor com exemplos adversariais
    """
    x_orig = x.clone().detach()
    adv_x = x.clone().detach()
    
    for step in range(steps):
        adv_x.requires_grad_(True)
        
        outputs = model(adv_x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        
        with torch.no_grad():
            adv_x = adv_x + alpha * adv_x.grad.sign()
            
            perturbation = torch.clamp(adv_x - x_orig, -eps, eps)
            adv_x = x_orig + perturbation
        
        adv_x = adv_x.detach()
    
    return adv_x


def cw_l2_attack(model, x, y, c=1.0, kappa=0, steps=100, lr=0.01):
    """
    Carlini & Wagner L2 attack (versão canônica).
    
    Implementa o ataque C&W usando mudança de variável com tanh
    para garantir que a perturbação seja minimizada enquanto
    causa misclassificação.
    
    Args:
        model: Modelo alvo
        x: Tensor com dados de entrada
        y: Tensor com labels
        c: Constante de regularização
        kappa: Margem de confiança
        steps: Número de iterações de otimização
        lr: Learning rate
        
    Returns:
        Tensor com exemplos adversariais
    """
    batch_size = x.size(0)
    
    x_clipped = torch.clamp(x, -0.999, 0.999)
    
    w = torch.atanh(x_clipped)
    w = w.clone().detach().requires_grad_(True)
    
    optimizer = optim.Adam([w], lr=lr)
    
    model.eval()
    
    best_adv = x.clone()
    best_l2 = torch.full((batch_size,), float('inf'))
    
    for step in range(steps):
        optimizer.zero_grad()
        
        adv_x = torch.tanh(w)
        
        outputs = model(adv_x)
        
        real_logits = outputs.gather(1, y.unsqueeze(1)).squeeze()
        
        other_logits = outputs.clone()
        other_logits.scatter_(1, y.unsqueeze(1), -1e10)
        max_other_logits = other_logits.max(1)[0]
        
        f_loss = torch.clamp(real_logits - max_other_logits + kappa, min=0)
        
        l2_dist = ((adv_x - x) ** 2).sum(dim=1).sqrt()
        
        loss = l2_dist.sum() + c * f_loss.sum()
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            misclassified = (pred != y)
            
            for i in range(batch_size):
                if misclassified[i] and l2_dist[i] < best_l2[i]:
                    best_l2[i] = l2_dist[i]
                    best_adv[i] = adv_x[i]
    
    return best_adv.detach()


# ============================================================================
# PARTE 3: AVALIAÇÃO E TRANSFERIBILIDADE
# ============================================================================

def evaluate_on_models(adv_X, y_true, class_names, models_dir, attack_type, eps_value):
    """
    Avalia exemplos adversariais em modelos treinados.
    
    Testa transferibilidade dos ataques entre diferentes arquiteturas
    de modelos (DecisionTree, RandomForest, MLP).
    
    Args:
        adv_X: Array numpy com exemplos adversariais
        y_true: Array numpy com labels verdadeiros
        class_names: Lista com nomes das classes
        models_dir: Diretório com modelos treinados
        attack_type: Tipo de ataque utilizado
        eps_value: Valor de epsilon utilizado
        
    Returns:
        DataFrame com métricas por modelo
    """
    models_to_test = ["DecisionTree", "RandomForest", "MLP"]
    
    results = []
    
    for model_name in models_to_test:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            logging.warning(f"Modelo não encontrado: {model_path}")
            continue
        
        logging.info(f"Avaliando {model_name}...")
        
        clf = joblib.load(model_path)
        y_pred = clf.predict(adv_X)
        
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        results.append({
            "Model": model_name,
            "Attack": attack_type,
            "Epsilon": eps_value,
            "Accuracy": acc,
            "Macro_Precision": report["macro avg"]["precision"],
            "Macro_Recall": report["macro avg"]["recall"],
            "Macro_F1": report["macro avg"]["f1-score"]
        })
        
        logging.info(f"  Accuracy: {acc:.4f}")
        logging.info(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
    
    return pd.DataFrame(results)


# ============================================================================
# PARTE 4: ANÁLISE DE PERTURBAÇÕES
# ============================================================================

def analyze_perturbations(X_clean, X_adv, feature_names, attack_type, eps, results_dir):
    """
    Analisa quais features foram mais perturbadas.
    
    Identifica vulnerabilidades específicas por feature, calculando
    estatísticas de perturbação (média, máximo, desvio padrão).
    
    Args:
        X_clean: Array com dados originais
        X_adv: Array com dados adversariais
        feature_names: Lista com nomes das features
        attack_type: Tipo de ataque
        eps: Valor de epsilon
        results_dir: Diretório de saída
        
    Returns:
        DataFrame com análise de perturbações
    """
    perturbations = np.abs(X_adv - X_clean)
    
    mean_pert = perturbations.mean(axis=0)
    max_pert = perturbations.max(axis=0)
    std_pert = perturbations.std(axis=0)
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Perturbation': mean_pert,
        'Max_Perturbation': max_pert,
        'Std_Perturbation': std_pert,
        'Attack': attack_type,
        'Epsilon': eps
    }).sort_values('Mean_Perturbation', ascending=False)
    
    output_path = os.path.join(results_dir, f"adv_perturbation_{attack_type}_eps{eps}.csv")
    df.to_csv(output_path, index=False)
    
    logging.info(f"Análise de perturbações salva: {output_path}")
    logging.info(f"Top 5 features mais perturbadas:")
    
    for _, row in df.head(5).iterrows():
        logging.info(f"  {row['Feature']}: {row['Mean_Perturbation']:.6f}")
    
    return df


def analyze_feature_vulnerability(results_dir):
    """
    Consolida análise de vulnerabilidade por feature.
    
    Combina resultados de todos os ataques para identificar
    features consistentemente vulneráveis.
    
    Args:
        results_dir: Diretório com arquivos de perturbação
        
    Returns:
        DataFrame com vulnerabilidade agregada por feature
    """
    logging.info("Analisando vulnerabilidade por feature")
    
    import glob
    pert_files = glob.glob(os.path.join(results_dir, "adv_perturbation_*.csv"))
    
    if not pert_files:
        logging.warning("Nenhum arquivo de perturbação encontrado")
        return pd.DataFrame()
    
    logging.info(f"Encontrados {len(pert_files)} arquivos de perturbação")
    
    all_dfs = []
    for f in pert_files:
        df = pd.read_csv(f)
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    feature_vuln = combined.groupby('Feature').agg({
        'Mean_Perturbation': 'mean',
        'Max_Perturbation': 'max',
        'Std_Perturbation': 'mean'
    }).reset_index().sort_values('Mean_Perturbation', ascending=False)
    
    top20 = feature_vuln.head(20)
    
    logging.info("\nTop 20 features mais vulneráveis:")
    for _, row in top20.head(10).iterrows():
        logging.info(f"  {row['Feature']}: {row['Mean_Perturbation']:.6f}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20['Mean_Perturbation'], color='steelblue', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20['Feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Perturbacao Media')
    ax.set_title('Top 20 Features Mais Vulneraveis a Ataques Adversariais')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    img_path = os.path.join(results_dir, "adv_feature_vulnerability.png")
    plt.savefig(img_path, dpi=160)
    plt.close()
    
    logging.info(f"Gráfico salvo: {img_path}")
    
    csv_path = os.path.join(results_dir, "adv_feature_vulnerability.csv")
    feature_vuln.to_csv(csv_path, index=False)
    logging.info(f"CSV salvo: {csv_path}")
    
    return feature_vuln


# ============================================================================
# PARTE 5: VISUALIZAÇÕES
# ============================================================================

def plot_comparison(original_metrics, adv_metrics, results_dir):
    """
    Plota comparação entre desempenho original e sob ataque adversarial.
    
    Args:
        original_metrics: DataFrame com métricas originais
        adv_metrics: DataFrame com métricas sob ataque
        results_dir: Diretório de saída
    """
    if original_metrics.empty or adv_metrics.empty:
        logging.warning("Métricas insuficientes para plotar comparação")
        return
    
    adv_agg = adv_metrics.groupby("Model")["Accuracy"].mean().reset_index()
    
    models = original_metrics["Model"].values
    orig_acc = original_metrics["Accuracy"].values * 100
    
    adv_acc = []
    for model in models:
        val = adv_agg[adv_agg["Model"] == model]["Accuracy"].values
        adv_acc.append(val[0] * 100 if len(val) > 0 else 0)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, orig_acc, width, label='Original', color='steelblue')
    bars2 = ax.bar(x + width/2, adv_acc, width, label='Adversarial (Media)', color='coral')
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Impacto de Ataques Adversariais na Acuracia')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    img_path = os.path.join(results_dir, "adv_comparison.png")
    plt.savefig(img_path, dpi=160)
    plt.close()
    
    logging.info(f"Gráfico de comparação salvo: {img_path}")


def plot_by_attack_type(adv_metrics, results_dir):
    """
    Plota comparação entre tipos de ataque.
    
    Args:
        adv_metrics: DataFrame com métricas adversariais
        results_dir: Diretório de saída
    """
    if adv_metrics.empty:
        logging.warning("Sem dados para plotar por tipo de ataque")
        return
    
    pivot = adv_metrics.pivot_table(
        values="Accuracy",
        index="Model",
        columns="Attack",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    pivot = pivot * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot.plot(kind='bar', ax=ax, alpha=0.8)
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustez por Tipo de Ataque Adversarial')
    ax.set_ylim([0, 105])
    ax.legend(title='Tipo de Ataque')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    img_path = os.path.join(results_dir, "adv_by_attack_type.png")
    plt.savefig(img_path, dpi=160)
    plt.close()
    
    logging.info(f"Gráfico por tipo de ataque salvo: {img_path}")


# ============================================================================
# PARTE 6: RELATÓRIO DE INSIGHTS
# ============================================================================

def generate_insights_report(results_dir, feature_vuln, adv_metrics):
    """
    Gera relatório textual com insights das análises.
    
    Args:
        results_dir: Diretório de saída
        feature_vuln: DataFrame com vulnerabilidade por feature
        adv_metrics: DataFrame com métricas adversariais
    """
    logging.info("Gerando relatório de insights")
    
    report = []
    
    report.append("="*70)
    report.append("ANALISE QUALITATIVA DOS ATAQUES ADVERSARIAIS")
    report.append("="*70)
    report.append("")
    
    if feature_vuln is not None and not feature_vuln.empty:
        report.append("1. VULNERABILIDADE POR FEATURE")
        report.append("-" * 70)
        report.append("")
        
        top5 = feature_vuln.head(5)
        
        report.append("As 5 features mais vulneraveis sao:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            report.append(f"  {i}. {row['Feature']}")
            report.append(f"     Perturbacao media: {row['Mean_Perturbation']:.6f}")
            report.append(f"     Perturbacao maxima: {row['Max_Perturbation']:.6f}")
            report.append("")
        
        report.append("INTERPRETACAO:")
        report.append("Features com alta perturbacao media sao mais suscetiveis a")
        report.append("manipulacao adversarial. Sistemas de defesa devem:")
        report.append("  - Monitorar essas features mais de perto")
        report.append("  - Aplicar regularizacao extra durante treinamento")
        report.append("  - Considerar feature engineering para reduzir sensibilidade")
        report.append("")
    
    if not adv_metrics.empty:
        report.append("2. PADROES DE MISCLASSIFICACAO")
        report.append("-" * 70)
        report.append("")
        
        model_robustness = adv_metrics.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
        
        report.append("ROBUSTEZ POR MODELO (Accuracy media sob ataque):")
        for model, acc in model_robustness.items():
            report.append(f"  {model}: {acc:.4f} ({acc*100:.1f}%)")
        
        report.append("")
        report.append("INTERPRETACAO:")
        
        most_robust = model_robustness.idxmax()
        if most_robust == 'RandomForest':
            report.append("Random Forest demonstrou maior robustez adversarial.")
            report.append("Motivo: Ensemble methods sao mais dificeis de enganar devido")
            report.append("a agregacao de multiplas arvores com diferentes decisoes.")
        elif most_robust == 'DecisionTree':
            report.append("Decision Tree demonstrou maior robustez adversarial.")
            report.append("Motivo: Modelos baseados em arvore usam thresholds discretos,")
            report.append("requerendo perturbacoes maiores para mudar predicao.")
        else:
            report.append("MLP (rede neural) demonstrou maior robustez adversarial.")
            report.append("Isso e incomum - geralmente redes neurais sao mais vulneraveis.")
        
        report.append("")
        
        attack_effectiveness = adv_metrics.groupby('Attack')['Accuracy'].mean().sort_values()
        
        report.append("EFETIVIDADE POR TIPO DE ATAQUE:")
        for attack, acc in attack_effectiveness.items():
            report.append(f"  {attack}: {acc:.4f} (menor = mais efetivo)")
        
        report.append("")
        report.append("INTERPRETACAO:")
        
        most_effective = attack_effectiveness.idxmin()
        if most_effective == 'CW_L2':
            report.append("C&W L2 foi o ataque mais efetivo, como esperado.")
            report.append("Motivo: C&W otimiza perturbacao minima que garante")
            report.append("misclassificacao, sendo mais sofisticado que FGSM/PGD.")
        elif most_effective == 'PGD':
            report.append("PGD foi o ataque mais efetivo.")
            report.append("Motivo: PGD usa multiplas iteracoes, sendo mais forte que FGSM.")
        else:
            report.append(f"{most_effective} foi o ataque mais efetivo.")
    
    report.append("")
    report.append("="*70)
    report.append("RECOMENDACOES PARA O ARTIGO")
    report.append("="*70)
    report.append("")
    report.append("Na Secao 4.3 (Resultados - Ataques Adversariais), incluir:")
    report.append("")
    report.append("1. Discutir features mais vulneraveis e implicacoes praticas")
    report.append("2. Comparar robustez dos modelos (RF > DT > MLP e padrao esperado)")
    report.append("3. Explicar por que C&W e mais efetivo que FGSM/PGD")
    report.append("4. Propor defesas: adversarial training, input transformation,")
    report.append("   feature squeezing")
    report.append("")
    report.append("="*70)
    
    report_path = os.path.join(results_dir, "adv_misclassification_patterns.txt")
    
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    logging.info(f"Relatorio de insights salvo: {report_path}")
    
    for line in report:
        logging.info(line)


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal de execução."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    logging.info("="*70)
    logging.info("SCRIPT 04: ATAQUES ADVERSARIAIS E ANALISE QUALITATIVA")
    logging.info("="*70)
    
    args = parse_args(config)
    
    eps_list = [float(x.strip()) for x in args.eps_list.split(",")]
    logging.info(f"Epsilons a testar: {eps_list}")
    
    attack_types = [x.strip().upper() for x in args.attack_types.split(",")]
    logging.info(f"Tipos de ataque: {attack_types}")
    
    models_dir = config["paths"]["models_dir"]
    results_dir = config["paths"]["results_dir"]
    
    splits_path = os.path.join(results_dir, "train_test_splits.pkl")
    if not os.path.exists(splits_path):
        logging.error("Splits não encontrados. Execute 02_train_evaluate.py primeiro")
        return 1
    
    logging.info(f"Carregando splits de {splits_path}...")
    X_train, X_test, y_train, y_test = joblib.load(splits_path)
    
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    le = joblib.load(le_path)
    class_names = list(le.classes_)
    
    import json
    with open(os.path.join(models_dir, "features_list.json"), "r") as f:
        feature_names = json.load(f)
    
    logging.info(f"Amostras teste: {len(X_test):,}")
    logging.info(f"Classes: {len(class_names)}")
    logging.info(f"Features: {len(feature_names)}")
    
    in_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    surrogate = SurrogateMLP(in_dim, n_classes)
    train_surrogate(surrogate, X_train, y_train, epochs=args.epochs, lr=args.lr)
    
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    surrogate.eval()
    
    with torch.no_grad():
        surr_outputs = surrogate(X_test_tensor)
        surr_preds = surr_outputs.argmax(dim=1).numpy()
        surr_acc = accuracy_score(y_test, surr_preds)
        logging.info(f"\nAccuracy do surrogate (dados limpos): {surr_acc:.4f}")
    
    all_adv_results = []
    all_perturbation_dfs = []
    
    for attack_type in attack_types:
        for eps in eps_list:
            logging.info("\n" + "="*70)
            logging.info(f"ATAQUE: {attack_type} | EPSILON: {eps}")
            logging.info("="*70)
            
            if attack_type == "FGSM":
                adv_X_tensor = fgsm_attack(surrogate, X_test_tensor, y_test_tensor, eps)
            
            elif attack_type == "PGD":
                adv_X_tensor = pgd_attack(
                    surrogate, X_test_tensor, y_test_tensor,
                    eps=eps, alpha=args.pgd_alpha, steps=args.pgd_steps
                )
            
            elif attack_type == "CW_L2":
                logging.info(f"Usando C&W L2 canonico (steps={args.cw_steps})")
                adv_X_tensor = cw_l2_attack(
                    surrogate, X_test_tensor, y_test_tensor,
                    c=1.0, kappa=0, steps=args.cw_steps, lr=args.cw_lr
                )
            
            else:
                logging.warning(f"Tipo de ataque desconhecido: {attack_type}")
                continue
            
            adv_X = adv_X_tensor.numpy()
            
            with torch.no_grad():
                adv_outputs = surrogate(adv_X_tensor)
                adv_preds = adv_outputs.argmax(dim=1).numpy()
                adv_acc_surr = accuracy_score(y_test, adv_preds)
                logging.info(f"Surrogate - Accuracy adversarial: {adv_acc_surr:.4f}")
                logging.info(f"Surrogate - Queda: {(surr_acc - adv_acc_surr)*100:.2f}%")
            
            pert_df = analyze_perturbations(
                X_test, adv_X, feature_names, attack_type, eps, results_dir
            )
            all_perturbation_dfs.append(pert_df)
            
            logging.info(f"\nAvaliando transferibilidade ({attack_type}, eps={eps})...")
            adv_df = evaluate_on_models(
                adv_X, y_test, class_names, models_dir, attack_type, eps
            )
            
            all_adv_results.append(adv_df)
    
    if not all_adv_results:
        logging.error("Nenhum resultado adversarial gerado")
        return 1
    
    all_adv_df = pd.concat(all_adv_results, ignore_index=True)
    
    detailed_path = os.path.join(results_dir, "adv_by_epsilon.csv")
    all_adv_df.to_csv(detailed_path, index=False)
    logging.info(f"\nResultados detalhados salvos: {detailed_path}")
    
    if all_perturbation_dfs:
        all_pert_df = pd.concat(all_perturbation_dfs, ignore_index=True)
        pert_path = os.path.join(results_dir, "adv_perturbation_analysis.csv")
        all_pert_df.to_csv(pert_path, index=False)
        logging.info(f"Analise de perturbacoes salva: {pert_path}")
    
    summary_df = all_adv_df.groupby("Model").agg({
        "Accuracy": "mean",
        "Macro_Precision": "mean",
        "Macro_Recall": "mean",
        "Macro_F1": "mean"
    }).reset_index()
    
    summary_df["Attack"] = "Media(FGSM+PGD+CW)"
    
    summary_path = os.path.join(results_dir, "adv_overall_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Metricas agregadas salvas: {summary_path}")
    
    logging.info("\n" + "="*70)
    logging.info("RESUMO - ROBUSTEZ ADVERSARIAL")
    logging.info("="*70)
    
    for _, row in summary_df.iterrows():
        logging.info(f"\n{row['Model']}:")
        logging.info(f"  Accuracy media: {row['Accuracy']:.4f}")
        logging.info(f"  Macro F1 medio: {row['Macro_F1']:.4f}")
    
    with open(os.path.join(results_dir, "adv_overall_table.tex"), "w") as f:
        latex_df = summary_df.copy()
        for col in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1"]:
            if col in latex_df.columns:
                latex_df[col] = (latex_df[col] * 100).round(2)
        
        f.write(latex_df.to_latex(
            index=False,
            caption="Metricas sob ataque adversarial (percentual)",
            label="tab:adv",
            escape=False
        ))
    
    orig_path = os.path.join(results_dir, "overall_metrics.csv")
    if os.path.exists(orig_path):
        orig_df = pd.read_csv(orig_path)
        plot_comparison(orig_df, all_adv_df, results_dir)
    
    plot_by_attack_type(all_adv_df, results_dir)
    
    feature_vuln = analyze_feature_vulnerability(results_dir)
    
    generate_insights_report(results_dir, feature_vuln, all_adv_df)
    
    logging.info("\n" + "="*70)
    logging.info("ATAQUES ADVERSARIAIS E ANALISE QUALITATIVA CONCLUIDOS")
    logging.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())