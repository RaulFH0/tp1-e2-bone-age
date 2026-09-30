"""
04_compare_models.py

Compara tres regressores classicos (SVR, Random Forest, Gradient Boosting)
sobre o mesmo descritor HOG, atendendo a exigencia da Semana 3: "ao menos
tres regressores classicos" (SVR, Random Forest e Gradient Boosting).

Extrai o HOG uma unica vez (com cache proprio, separado do cache do script
03) e treina os tres modelos sobre as mesmas features, reportando MAE/RMSE/R2
de cada um no conjunto de validacao, junto com a baseline da media do treino.

Uso:
    python 04_compare_models.py \
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

# Reaproveita extract_features (HOG + sexo) do script 02, que ja usa
# preprocess_image (224x224) e o split/preprocessamento oficiais da equipe.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "hog_baseline", Path(__file__).parent / "02_hog_baseline.py"
)
_hog_baseline = importlib.util.module_from_spec(_spec)
sys.modules["hog_baseline"] = _hog_baseline
_spec.loader.exec_module(_hog_baseline)

RANDOM_SEED = 42

# Parametros "de bom senso" para cada modelo -- ainda nao passaram por busca
MODELS = {
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED
    ),
}


def get_features(images_dir: Path, csv_path: str, splits_dir: Path, cache_path: Path):
    if cache_path.exists():
        print(f"Reaproveitando cache de features em {cache_path}")
        cache = joblib.load(cache_path)
        return cache["X_train"], cache["y_train"], cache["X_val"], cache["y_val"]

    print("Cache nao encontrado -- extraindo HOG (split/preprocess oficiais)...")
    meta = pd.read_csv(csv_path)
    train_ids = pd.read_csv(splits_dir / "train_ids.csv")["id"].tolist()
    val_ids = pd.read_csv(splits_dir / "val_ids.csv")["id"].tolist()
    meta_idx = meta.set_index("id")
    y_train = meta_idx.loc[train_ids, "boneage"].values
    y_val = meta_idx.loc[val_ids, "boneage"].values

    X_train = _hog_baseline.extract_features(train_ids, images_dir, meta)
    X_val = _hog_baseline.extract_features(val_ids, images_dir, meta)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val},
        cache_path,
    )
    print(f"Features cacheadas em {cache_path} (proximas execucoes serao rapidas)")
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
    # cache proprio (nao mistura com o do script 03, que usa outra amostra)
    cache_path = out_dir / "features_cache_hog_oficial.joblib"

    X_train, y_train, X_val, y_val = get_features(images_dir, csv_path, splits_dir, cache_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit SOMENTE no treino
    X_val_scaled = scaler.transform(X_val)

    baseline_pred = np.full_like(y_val, fill_value=y_train.mean(), dtype=float)
    results = {"media_treino": evaluate(y_val, baseline_pred)}

    for name, model in MODELS.items():
        print(f"Treinando {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        results[name] = evaluate(y_val, y_pred)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_hog_comparacao_modelos.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Comparacao de modelos (HOG, validacao) ===")
    print(f"{'Modelo':<18} {'MAE (meses)':>12} {'RMSE (meses)':>14} {'R2':>8}")
    for name, m in results.items():
        print(f"{name:<18} {m['MAE_meses']:>12.2f} {m['RMSE_meses']:>14.2f} {m['R2']:>8.3f}")

    print(f"\nSalvo em {out_dir / 'metrics_hog_comparacao_modelos.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.images_dir, args.csv, Path(args.splits_dir), args.out_dir)
