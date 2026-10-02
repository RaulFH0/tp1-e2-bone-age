"""
06_error_analysis.py

CONTINGENCIA: protocolo, metricas e analise de erro eram responsabilidade da
Rilari, conforme o plano original da equipe. Como ela nao respondeu aos
pedidos de status as vesperas da entrega, Raul implementou esta versao para
nao travar o artigo. Se a Rilari enviar sua propria analise antes da
entrega, ela deve revisar/substituir este script.

Gera, para o MELHOR modelo encontrado ate agora (HOG + Gradient Boosting):
    1. Grafico de Bland-Altman (media vs. diferenca entre previsto e real)
    2. MAE por faixa etaria (bins de 24 meses), exigido pelo enunciado
    3. Dispersao real x previsto

Saida: results/bland_altman.png, results/erro_por_faixa_etaria.png,
results/real_vs_previsto.png, results/analise_erro.json

Uso:
    python 06_error_analysis.py \
        --images-dir "data/raw/boneage-training-dataset/boneage-training-dataset" \
        --csv data/raw/boneage-training-dataset.csv \
        --splits-dir data/splits \
        --out-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "hog_baseline", Path(__file__).parent / "02_hog_baseline.py"
)
_hog_baseline = importlib.util.module_from_spec(_spec)
sys.modules["hog_baseline"] = _hog_baseline
_spec.loader.exec_module(_hog_baseline)

RANDOM_SEED = 42
AGE_BIN_WIDTH_MONTHS = 24


def get_hog_features(images_dir: Path, csv_path: str, splits_dir: Path, out_dir: Path):
    cache_path = out_dir / "features_cache_hog_oficial.joblib"
    if cache_path.exists():
        print(f"Reaproveitando cache de HOG em {cache_path}")
        cache = joblib.load(cache_path)
        return cache["X_train"], cache["y_train"], cache["X_val"], cache["y_val"]

    print("Cache de HOG nao encontrado -- extraindo (mesma demora de sempre)...")
    meta = pd.read_csv(csv_path)
    train_ids = pd.read_csv(splits_dir / "train_ids.csv")["id"].tolist()
    val_ids = pd.read_csv(splits_dir / "val_ids.csv")["id"].tolist()
    meta_idx = meta.set_index("id")
    y_train = meta_idx.loc[train_ids, "boneage"].values
    y_val = meta_idx.loc[val_ids, "boneage"].values
    X_train = _hog_baseline.extract_features(train_ids, images_dir, meta)
    X_val = _hog_baseline.extract_features(val_ids, images_dir, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}, cache_path)
    return X_train, y_train, X_val, y_val


def plot_bland_altman(y_true, y_pred, out_path: Path):
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    bias = diff_vals.mean()
    sd = diff_vals.std()
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    plt.figure(figsize=(7, 5))
    plt.scatter(mean_vals, diff_vals, alpha=0.4, s=15)
    plt.axhline(bias, color="red", linestyle="-", label=f"Viés médio = {bias:.1f} meses")
    plt.axhline(upper, color="gray", linestyle="--", label=f"+1,96 DP = {upper:.1f}")
    plt.axhline(lower, color="gray", linestyle="--", label=f"-1,96 DP = {lower:.1f}")
    plt.xlabel("Média entre idade real e prevista (meses)")
    plt.ylabel("Diferença: previsto - real (meses)")
    plt.title("Gráfico de Bland-Altman — HOG + Gradient Boosting (validação)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return {"vies_medio": float(bias), "limite_superior_95": float(upper), "limite_inferior_95": float(lower)}


def plot_scatter_real_vs_previsto(y_true, y_pred, out_path: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, color="red", linestyle="--", label="Previsão perfeita")
    plt.xlabel("Idade óssea real (meses)")
    plt.ylabel("Idade óssea prevista (meses)")
    plt.title("Real vs. Previsto — HOG + Gradient Boosting (validação)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def mae_por_faixa_etaria(y_true, y_pred, out_path: Path):
    age_bin = (y_true // AGE_BIN_WIDTH_MONTHS).astype(int)
    df = pd.DataFrame({"age_bin": age_bin, "y_true": y_true, "y_pred": y_pred})
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    grouped = df.groupby("age_bin").agg(
        n=("abs_err", "size"), mae=("abs_err", "mean"), idade_media=("y_true", "mean")
    ).reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["idade_media"], grouped["mae"], width=AGE_BIN_WIDTH_MONTHS * 0.8)
    plt.xlabel("Idade óssea real média da faixa (meses)")
    plt.ylabel("MAE (meses)")
    plt.title("Erro por faixa etária (bins de 24 meses) — HOG + Gradient Boosting")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return grouped.to_dict(orient="records")


def main(images_dir: str, csv_path: str, splits_dir: str, out_dir: str):
    images_dir = Path(images_dir)
    splits_dir = Path(splits_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = get_hog_features(images_dir, csv_path, splits_dir, out_dir)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED
    )
    print("Treinando Gradient Boosting (HOG) para análise de erro...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)

    ba_stats = plot_bland_altman(y_val, y_pred, out_dir / "bland_altman.png")
    plot_scatter_real_vs_previsto(y_val, y_pred, out_dir / "real_vs_previsto.png")
    faixa_etaria = mae_por_faixa_etaria(y_val, y_pred, out_dir / "erro_por_faixa_etaria.png")

    result = {
        "_nota": (
            "Analise de erro implementada por Raul como contingencia "
            "(Rilari nao respondeu). Revisar/substituir se ela enviar versao propria."
        ),
        "modelo": "HOG + Gradient Boosting",
        "bland_altman": ba_stats,
        "erro_por_faixa_etaria": faixa_etaria,
    }
    with open(out_dir / "analise_erro.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== Análise de erro concluída ===")
    print(f"Viés médio (Bland-Altman): {ba_stats['vies_medio']:.2f} meses")
    print("MAE por faixa etária:")
    for row in faixa_etaria:
        print(f"  ~{row['idade_media']:.0f} meses (n={row['n']}): MAE {row['mae']:.2f}")
    print(f"\nGráficos salvos em {out_dir}/ (bland_altman.png, real_vs_previsto.png, erro_por_faixa_etaria.png)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.images_dir, args.csv, Path(args.splits_dir), args.out_dir)
