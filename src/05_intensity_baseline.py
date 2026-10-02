"""
05_intensity_baseline.py

CONTINGENCIA: esta familia de descritores (intensidade) era responsabilidade
da Rilari, conforme o plano original da equipe. Como ela nao respondeu aos
pedidos de status as vesperas da entrega, Raul implementou esta versao para
nao travar a comparacao final. Os parametros abaixo
sao uma escolha razoavel, mas nao foram validados/discutidos com ela.

Extrai um descritor de intensidade sobre a MESMA imagem pre-processada
(preprocess_image: 224x224, escala de cinza, normalizada em [0,1]) usada por
HOG e textura, e treina os mesmos tres regressores classicos (SVR, Random
Forest, Gradient Boosting) para manter a comparacao justa entre familias.

Descritor de intensidade (parametros documentados para o artigo):
    - histograma de intensidade de pixel, 32 bins, intervalo [0,1]
    - media e desvio-padrao da intensidade de todos os pixels
    - percentis 25, 50 (mediana) e 75
  Total: 32 + 2 + 3 = 37 valores, + sexo = 38 caracteristicas por imagem.

Uso:
    python 05_intensity_baseline.py \
        --images-dir "data/raw/boneage-training-dataset/boneage-training-dataset" \
        --csv data/raw/boneage-training-dataset.csv \
        --splits-dir data/splits \
        --out-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from image_preprocessing import preprocess_image  # funcao comum da equipe

RANDOM_SEED = 42
N_HIST_BINS = 32

MODELS = {
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED
    ),
}


def extract_intensity_features(ids, images_dir: Path, meta: pd.DataFrame) -> np.ndarray:
    """Histograma de intensidade (32 bins) + media/desvio/percentis + sexo."""
    feats = []
    meta_idx = meta.set_index("id")
    for img_id in ids:
        img_path = images_dir / f"{img_id}.png"
        img = preprocess_image(img_path)  # float32, 224x224, [0,1]
        pixels = img.ravel()

        hist, _ = np.histogram(pixels, bins=N_HIST_BINS, range=(0.0, 1.0), density=True)
        mean_std = [pixels.mean(), pixels.std()]
        percentiles = np.percentile(pixels, [25, 50, 75])
        sex = float(meta_idx.loc[img_id, "male"])

        feats.append(np.concatenate([hist, mean_std, percentiles, [sex]]))
    return np.vstack(feats)


def get_features(images_dir: Path, csv_path: str, splits_dir: Path, cache_path: Path):
    if cache_path.exists():
        print(f"Reaproveitando cache de features em {cache_path}")
        cache = joblib.load(cache_path)
        return cache["X_train"], cache["y_train"], cache["X_val"], cache["y_val"]

    print("Cache nao encontrado -- extraindo intensidade (pode demorar)...")
    meta = pd.read_csv(csv_path)
    train_ids = pd.read_csv(splits_dir / "train_ids.csv")["id"].tolist()
    val_ids = pd.read_csv(splits_dir / "val_ids.csv")["id"].tolist()
    meta_idx = meta.set_index("id")
    y_train = meta_idx.loc[train_ids, "boneage"].values
    y_val = meta_idx.loc[val_ids, "boneage"].values

    X_train = extract_intensity_features(train_ids, images_dir, meta)
    X_val = extract_intensity_features(val_ids, images_dir, meta)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val},
        cache_path,
    )
    print(f"Features cacheadas em {cache_path}")
    return X_train, y_train, X_val, y_val


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE_meses": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_meses": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main(images_dir: str, csv_path: str, splits_dir: str, out_dir: str):
    images_dir = Path(images_dir)
    splits_dir = Path(splits_dir)
    out_dir = Path(out_dir)
    cache_path = out_dir / "features_cache_intensidade.joblib"

    X_train, y_train, X_val, y_val = get_features(images_dir, csv_path, splits_dir, cache_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    baseline_pred = np.full_like(y_val, fill_value=y_train.mean(), dtype=float)
    results = {"media_treino": evaluate(y_val, baseline_pred)}

    for name, model in MODELS.items():
        print(f"Treinando {name} (intensidade)...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        results[name] = evaluate(y_val, y_pred)

    results["_nota"] = (
        "Descritor de intensidade implementado por Raul como contingencia "
        "(Rilari nao respondeu). Parametros: histograma 32 bins + media/desvio/"
        "percentis 25-50-75 + sexo."
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_intensidade_comparacao_modelos.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Comparacao de modelos (Intensidade, validacao) ===")
    print(f"{'Modelo':<18} {'MAE (meses)':>12} {'RMSE (meses)':>14} {'R2':>8}")
    for name, m in results.items():
        if name == "_nota":
            continue
        print(f"{name:<18} {m['MAE_meses']:>12.2f} {m['RMSE_meses']:>14.2f} {m['R2']:>8.3f}")

    print(f"\nSalvo em {out_dir / 'metrics_intensidade_comparacao_modelos.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.images_dir, args.csv, Path(args.splits_dir), args.out_dir)
