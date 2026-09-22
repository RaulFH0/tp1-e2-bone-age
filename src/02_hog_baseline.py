"""
02_hog_baseline.py

Extrai descritores HOG das radiografias e treina/avalia um primeiro regressor
(SVR) para a idade ossea em meses. Cobre a entrega da Semana 2 da parte do
Raul: "1 descritor + 1 modelo".

Uso tipico:
    python 02_hog_baseline.py \
        --images-dir data/raw/boneage-train-dataset/boneage-train-dataset \
        --csv data/raw/boneage-train-dataset.csv \
        --splits-dir data/splits \
        --out-dir results

Parametros do HOG documentados abaixo (registrar no artigo, secao Metodologia):
    - imagem redimensionada para IMG_SIZE x IMG_SIZE (grayscale)
    - orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2)
    - feature_vector=True

O sexo (coluna "male") entra como covariavel explicita, concatenada ao vetor HOG.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

IMG_SIZE = 256
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    feature_vector=True,
)
RANDOM_SEED = 42


def load_image_gray(path: Path) -> np.ndarray:
    img = imread(path, as_gray=True)
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    return img


def extract_features(ids, images_dir: Path, meta: pd.DataFrame) -> np.ndarray:
    """Extrai HOG + covariavel sexo para uma lista de ids."""
    feats = []
    meta_idx = meta.set_index("id")
    for img_id in ids:
        # Ajuste a extensao (.png) conforme o formato real dos arquivos baixados
        img_path = images_dir / f"{img_id}.png"
        img = load_image_gray(img_path)
        hog_feat = hog(img, **HOG_PARAMS)
        sex = float(meta_idx.loc[img_id, "male"])
        feats.append(np.concatenate([hog_feat, [sex]]))
    return np.vstack(feats)


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
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(csv_path)
    train_ids = pd.read_csv(splits_dir / "train_ids.csv")["id"].tolist()
    val_ids = pd.read_csv(splits_dir / "val_ids.csv")["id"].tolist()

    meta_idx = meta.set_index("id")
    y_train = meta_idx.loc[train_ids, "boneage"].values
    y_val = meta_idx.loc[val_ids, "boneage"].values

    print(f"Extraindo HOG de {len(train_ids)} imagens de treino...")
    t0 = time.time()
    X_train = extract_features(train_ids, images_dir, meta)
    print(f"  concluido em {time.time() - t0:.1f}s")

    print(f"Extraindo HOG de {len(val_ids)} imagens de validacao...")
    X_val = extract_features(val_ids, images_dir, meta)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit SOMENTE no treino
    X_val_scaled = scaler.transform(X_val)

    print("Treinando SVR (kernel RBF, C=10, epsilon=1)...")
    model = SVR(kernel="rbf", C=10.0, epsilon=1.0)
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)

    # Baseline trivial para referencia local (o oficial e responsabilidade da Rilari)
    baseline_pred_val = np.full_like(y_val, fill_value=y_train.mean(), dtype=float)

    metrics = {
        "svr_hog_train": evaluate(y_train, y_pred_train),
        "svr_hog_val": evaluate(y_val, y_pred_val),
        "media_treino_val": evaluate(y_val, baseline_pred_val),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "hog_params": HOG_PARAMS,
        "img_size": IMG_SIZE,
        "random_seed": RANDOM_SEED,
    }

    with open(out_dir / "metrics_hog_svr.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Resultados (validacao) ===")
    print(f"Media do treino  -> MAE: {metrics['media_treino_val']['MAE_meses']:.2f} meses")
    print(f"HOG + SVR        -> MAE: {metrics['svr_hog_val']['MAE_meses']:.2f} meses")
    print(f"\nSalvo em {out_dir / 'metrics_hog_svr.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.images_dir, args.csv, args.splits_dir, args.out_dir)
