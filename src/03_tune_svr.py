"""
03_tune_svr.py

Ajuste rapido de hiperparametros do SVR sobre as features HOG.

Na PRIMEIRA execucao, extrai o HOG de novo (mesma demora do script 02) e
SALVA em cache (results/features_cache.joblib). Nas execucoes seguintes,
reaproveita o cache.

A busca de hiperparametros roda em DUAS etapas para ficar rapida:
  1) RandomizedSearchCV (poucas combinacoes, poucas dobras) numa AMOSTRA
     de ate SEARCH_SAMPLE_SIZE exemplos do treino -- SVR com kernel RBF
     escala mal com o numero de amostras, entao buscar na base inteira
     (milhares de imagens) pode travar por horas.
  2) O modelo final, com os melhores parametros encontrados, e treinado
     no conjunto de treino COMPLETO (isso sim demora um pouco, mas e um
     unico treino, nao dezenas).

Uso:
    python 03_tune_svr.py \
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# Reaproveita as mesmas funcoes de extracao do script 02
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "hog_baseline", Path(__file__).parent / "02_hog_baseline.py"
)
_hog_baseline = importlib.util.module_from_spec(_spec)
sys.modules["hog_baseline"] = _hog_baseline
_spec.loader.exec_module(_hog_baseline)

RANDOM_SEED = 42

# Grid pequeno de proposito para ser rapido; expanda depois se quiser.
# C=100 e gamma='auto' tendem a demorar MUITO para convergir com poucos
# milhares de amostras e kernel RBF -- por isso ficamos numa faixa mais
# comportada e usamos max_iter para nunca deixar uma combinacao travar sozinha.
PARAM_DIST = {
    "svr__C": [1, 5, 10, 25],
    "svr__epsilon": [0.5, 1.0, 2.0],
    "svr__gamma": ["scale"],
}
N_ITER_SEARCH = 8       # numero de combinacoes testadas (RandomizedSearchCV)
N_CV_FOLDS = 3          # menos dobras = mais rapido
SEARCH_SAMPLE_SIZE = 2500  # busca de hiperparametro roda numa amostra, nao no treino inteiro
SVR_MAX_ITER = 5000     # teto de iteracoes -- evita travar sem convergir


def get_features(images_dir: Path, csv_path: str, splits_dir: Path, cache_path: Path):
    if cache_path.exists():
        print(f"Reaproveitando cache de features em {cache_path}")
        cache = joblib.load(cache_path)
        return cache["X_train"], cache["y_train"], cache["X_val"], cache["y_val"]

    print("Cache nao encontrado — extraindo HOG (mesma demora do script 02)...")
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


def main(images_dir: str, csv_path: str, splits_dir: str, out_dir: str):
    images_dir = Path(images_dir)
    splits_dir = Path(splits_dir)
    out_dir = Path(out_dir)
    cache_path = out_dir / "features_cache.joblib"

    X_train, y_train, X_val, y_val = get_features(images_dir, csv_path, splits_dir, cache_path)

    # ETAPA 1: busca de hiperparametro numa AMOSTRA do treino (rapido).
    rng = np.random.RandomState(RANDOM_SEED)
    n_search = min(SEARCH_SAMPLE_SIZE, X_train.shape[0])
    sample_idx = rng.choice(X_train.shape[0], size=n_search, replace=False)
    X_search = X_train[sample_idx]
    y_search = y_train[sample_idx]

    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("svr", SVR(kernel="rbf", max_iter=SVR_MAX_ITER))]),
        PARAM_DIST,
        n_iter=N_ITER_SEARCH,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    print(f"Etapa 1/2: buscando hiperparametros em amostra de {n_search} exemplos "
          f"({N_ITER_SEARCH} combinacoes x {N_CV_FOLDS} dobras -- rapido)...")
    search.fit(X_search, y_search)
    print(f"Melhores parametros na amostra: {search.best_params_}")

    # ETAPA 2: treina o modelo final, com os melhores parametros, no treino INTEIRO.
    print("Etapa 2/2: treinando modelo final no conjunto de treino completo...")
    best_model = Pipeline([("scaler", StandardScaler()),
                           ("svr", SVR(kernel="rbf", max_iter=SVR_MAX_ITER))])
    best_model.set_params(**search.best_params_)
    best_model.fit(X_train, y_train)
    y_pred_val = best_model.predict(X_val)

    metrics = {
        "best_params": search.best_params_,
        "cv_mae_amostra_busca": float(-search.best_score_),
        "n_amostra_busca": n_search,
        "val_MAE_meses": float(mean_absolute_error(y_val, y_pred_val)),
        "val_RMSE_meses": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        "val_R2": float(r2_score(y_val, y_pred_val)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_hog_svr_tuned.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Melhor configuracao encontrada ===")
    print(f"Parametros: {metrics['best_params']}")
    print(f"MAE na busca (amostra de {n_search}): {metrics['cv_mae_amostra_busca']:.2f} meses")
    print(f"MAE validacao (modelo final, treino completo): {metrics['val_MAE_meses']:.2f} meses")
    print("\nCompare somente com resultados da mesma partição e pré-processamento.")
    print(f"Salvo em {out_dir / 'metrics_hog_svr_tuned.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.images_dir, args.csv, Path(args.splits_dir), args.out_dir)
