#!/usr/bin/env python3
"""
Script 00: Validação do Ambiente de Execução

Descrição:
    Verifica a integridade e disponibilidade de todos os componentes necessários
    para execução do pipeline de análise de segurança cibernética.

Validações realizadas:
    - Versão do interpretador Python
    - Dependências Python instaladas
    - Arquivo de configuração (config.yaml)
    - Estrutura de diretórios do dataset CIC-IDS-2017
    - Ferramentas de sistema (Suricata, tcpreplay, tshark)

Uso:
    python3 00_sanity_check.py

Autor: Bernardo Vivian Vieira
Data: Outubro 2025
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


def check_python_version():
    """
    Valida versão mínima do Python (3.8+).
    
    Returns:
        bool: True se versão adequada, False caso contrário
    """
    print("Verificando versão do Python...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERRO: Python {version.major}.{version.minor} detectado.")
        print("       Necessário Python >= 3.8")
        return False
    
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """
    Verifica instalação das dependências Python requeridas.
    
    Returns:
        bool: True se todas dependências instaladas, False caso contrário
    """
    print("\nVerificando dependências Python...")
    
    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "joblib": "joblib",
        "matplotlib": "matplotlib",
        "yaml": "pyyaml"
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  OK: {package}")
        except ImportError:
            print(f"  ERRO: {package} não encontrado")
            missing.append(package)
    
    if missing:
        print(f"\nPacotes ausentes: {', '.join(missing)}")
        print(f"Instale com: pip install {' '.join(missing)}")
        return False
    
    print("OK: Todas dependências instaladas")
    return True


def check_config_file():
    """
    Valida existência e estrutura do arquivo de configuração.
    
    Returns:
        bool: True se config.yaml válido, False caso contrário
    """
    print("\nVerificando arquivo de configuração...")
    
    if not os.path.exists("config.yaml"):
        print("ERRO: config.yaml não encontrado")
        print("      Crie o arquivo na raiz do projeto")
        return False
    
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        required_keys = ["random_seed", "paths", "models", "adversarial"]
        missing_keys = [k for k in required_keys if k not in config]
        
        if missing_keys:
            print(f"ERRO: Chaves ausentes no config.yaml: {', '.join(missing_keys)}")
            return False
        
        required_paths = [
            "data_dir", "csv_dir", "pcap_dir", 
            "models_dir", "results_dir", "runs_dir", "combined_csv"
        ]
        missing_paths = [p for p in required_paths if p not in config.get("paths", {})]
        
        if missing_paths:
            print(f"ERRO: Subchaves ausentes em 'paths': {', '.join(missing_paths)}")
            return False
        
        required_models = ["DecisionTree", "RandomForest", "MLP"]
        missing_models = [m for m in required_models if m not in config.get("models", {})]
        
        if missing_models:
            print(f"ERRO: Modelos ausentes em 'models': {', '.join(missing_models)}")
            return False
        
        print("OK: config.yaml válido")
        return True
        
    except yaml.YAMLError as e:
        print(f"ERRO: Falha ao ler config.yaml: {e}")
        return False
    except Exception as e:
        print(f"ERRO: Validação de config.yaml falhou: {e}")
        return False


def check_dataset(config):
    """
    Verifica presença e estrutura do dataset CIC-IDS-2017.
    
    Args:
        config (dict): Dicionário de configuração carregado
        
    Returns:
        bool: True se dataset válido, False caso contrário
    """
    print("\nVerificando dataset CIC-IDS-2017...")
    
    data_dir = config["paths"]["data_dir"]
    csv_dir = config["paths"]["csv_dir"]
    pcap_dir = config["paths"]["pcap_dir"]
    
    checks = [
        (data_dir, "Diretório principal do dataset"),
        (csv_dir, "Diretório de arquivos CSV"),
        (pcap_dir, "Diretório de arquivos PCAP")
    ]
    
    all_ok = True
    for path, description in checks:
        if os.path.exists(path):
            print(f"  OK: {description}: {path}")
        else:
            print(f"  ERRO: {description} não encontrado: {path}")
            all_ok = False
    
    if all_ok:
        csv_path = Path(csv_dir)
        pcap_path = Path(pcap_dir)
        
        csv_count = len(list(csv_path.glob("*.csv"))) if csv_path.exists() else 0
        pcap_count = (
            len(list(pcap_path.glob("*.pcap"))) + 
            len(list(pcap_path.glob("*.pcapng")))
        ) if pcap_path.exists() else 0
        
        print(f"  Arquivos CSV encontrados: {csv_count}")
        print(f"  Arquivos PCAP encontrados: {pcap_count}")
        
        if csv_count == 0:
            print(f"  AVISO: Nenhum arquivo CSV encontrado em {csv_dir}")
            all_ok = False
        
        if pcap_count == 0:
            print(f"  AVISO: Nenhum arquivo PCAP encontrado")
            print(f"         Scripts de integração IDS não poderão executar")
    
    if not all_ok:
        print("\nSOLUCAO:")
        print("  1. Baixe CIC-IDS-2017 de:")
        print("     https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  2. Extraia os arquivos")
        print("  3. Ajuste os caminhos em config.yaml")
    else:
        print("OK: Dataset presente e válido")
    
    return all_ok


def check_directories(config):
    """
    Cria diretórios de trabalho necessários se não existirem.
    
    Args:
        config (dict): Dicionário de configuração carregado
        
    Returns:
        bool: True (sempre bem-sucedido)
    """
    print("\nVerificando diretórios de trabalho...")
    
    dirs_to_create = [
        config["paths"]["models_dir"],
        config["paths"]["results_dir"],
        config["paths"]["runs_dir"],
        "data"
    ]
    
    for directory in dirs_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  OK: {directory}/")
    
    print("OK: Diretórios prontos")
    return True


def check_system_tools():
    """
    Verifica disponibilidade de ferramentas de sistema (opcional).
    
    Returns:
        bool: True (não bloqueia execução se ausentes)
    """
    print("\nVerificando ferramentas de sistema (opcional)...")
    
    tools = {
        "suricata": "Suricata IDS",
        "tcpreplay": "Replay de tráfego de rede",
        "tshark": "Análise de pacotes (Wireshark CLI)"
    }
    
    found = []
    missing = []
    needs_sudo = []
    
    for cmd, description in tools.items():
        found_tool = False
        
        try:
            result = subprocess.run(
                [cmd, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  OK: {description} ({cmd})")
                found.append(cmd)
                found_tool = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        if not found_tool and cmd == "suricata":
            common_paths = [
                "/usr/bin/suricata",
                "/usr/local/bin/suricata",
                "/usr/sbin/suricata",
                "/opt/suricata/bin/suricata"
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    try:
                        result = subprocess.run(
                            ["sudo", "-n", path, "-V"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=5
                        )
                        if result.returncode == 0:
                            version = result.stdout.decode().split('\n')[0]
                            print(f"  OK: {description} encontrado em: {path}")
                            print(f"      Versão: {version}")
                            print(f"      Requer sudo - configure config.yaml com 'binary: sudo suricata'")
                            needs_sudo.append(cmd)
                            found_tool = True
                            break
                    except (subprocess.TimeoutExpired, PermissionError):
                        print(f"  OK: {description} encontrado em: {path}")
                        print(f"      Requer sudo")
                        needs_sudo.append(cmd)
                        found_tool = True
                        break
        
        if not found_tool:
            print(f"  AVISO: {description} ({cmd}) não encontrado")
            missing.append(cmd)
    
    if missing:
        print(f"\nFerramentas ausentes: {', '.join(missing)}")
        if "suricata" in missing:
            print("  Instalação do Suricata:")
            print("    Ubuntu/Debian: sudo apt-get install suricata")
            print("    Fedora: sudo dnf install suricata")
            print("    macOS: brew install suricata")
            print("    OU execute: ./run_all.sh --skip-suricata")
    
    if needs_sudo and not missing:
        print(f"\nFerramentas que requerem sudo: {', '.join(needs_sudo)}")
    elif not missing and not needs_sudo:
        print("OK: Todas ferramentas de sistema disponíveis")
    
    return True


def main():
    """
    Função principal de validação.
    
    Returns:
        int: 0 se validação bem-sucedida, 1 caso contrário
    """
    print("=" * 70)
    print("VALIDACAO DO AMBIENTE DE EXECUCAO")
    print("TCC: Inteligência Artificial na Segurança Cibernética")
    print("=" * 70)
    
    config = None
    if os.path.exists("config.yaml"):
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"AVISO: Erro ao carregar config.yaml: {e}")
    
    checks = [
        ("Versão Python", check_python_version),
        ("Dependências", check_dependencies),
        ("Arquivo config.yaml", check_config_file),
    ]
    
    if config:
        checks.extend([
            ("Dataset", lambda: check_dataset(config)),
            ("Diretórios", lambda: check_directories(config)),
        ])
    
    checks.append(("Ferramentas do sistema", check_system_tools))
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func() if callable(check_func) else check_func
            results.append((name, result))
        except Exception as e:
            print(f"\nERRO ao verificar {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("RESUMO DA VALIDACAO")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "OK" if result else "FALHA"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 70)
    
    if passed == total:
        print("VALIDACAO CONCLUIDA COM SUCESSO")
        print("\nPróximos passos:")
        print("  1. Execute: python3 01_preprocessing.py")
        print("  2. OU execute pipeline completo: ./run_all.sh")
        print("=" * 70)
        return 0
    else:
        print(f"VALIDACAO FALHOU: {total - passed}/{total} verificações falharam")
        print("Corrija os problemas acima antes de prosseguir")
        
        failed_checks = [name for name, result in results if not result]
        print("\nSugestões:")
        
        if "Dependências" in failed_checks:
            print("  - pip install -r requirements.txt")
        
        if "Dataset" in failed_checks:
            print("  - Baixe CIC-IDS-2017 e ajuste config.yaml")
        
        if "Arquivo config.yaml" in failed_checks:
            print("  - Crie config.yaml baseado no exemplo do README.md")
        
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())