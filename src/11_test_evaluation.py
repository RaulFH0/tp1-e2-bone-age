"""
11_test_evaluation.py

AVALIACAO FINAL UNICA no conjunto de teste (600 imagens), para as tres
familias de descritores (HOG, textura LBP+GLCM, intensidade) x tres
modelos classicos (SVR, Random Forest, Gradient Boosting).



Uso:
    python 11_test_evaluation.py \
        --images-dir "data/raw/boneage-training-dataset/boneage-training-dataset" \
        --csv data/raw/boneage-training-dataset.csv \
        --splits-dir data/splits \
        --texture-dir data/textura_daniel \
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

import importlib.util
import sys

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

HERE = Path(__file__).parent
_hog = _load_module("hog_baseline", HERE / "02_hog_baseline.py")
_inten = _load_module("intensity_baseline", HERE / "05_intensity_baseline.py")

RANDOM_SEED = 42

MODEL_CONFIGS = lambda: {
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=1.0, gamma="scale"),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED
    ),
}


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE_meses": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_meses": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_family(name, X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # fit SOMENTE no treino
    X_test_s = scaler.transform(X_test)

    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    results = {"media_treino": evaluate(y_test, baseline_pred)}

    for model_name, model in MODEL_CONFIGS().items():
        print(f"  Treinando {model_name} ({name})...")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results[model_name] = evaluate(y_test, y_pred)

    return results


def get_hog(images_dir, csv_path, splits_dir, out_dir, train_ids, test_ids, meta):
    cache_path = out_dir / "features_cache_hog_oficial.joblib"
    if cache_path.exists():
        print("Reaproveitando cache de HOG (treino) ja existente...")
        cache = joblib.load(cache_path)
        X_train, y_train = cache["X_train"], cache["y_train"]
    else:
        print("Extraindo HOG do treino (sem cache)...")
        meta_idx = meta.set_index("id")
        y_train = meta_idx.loc[train_ids, "boneage"].values
        X_train = _hog.extract_features(train_ids, images_dir, meta)

    print("Extraindo HOG do TESTE (600 imagens, execucao unica)...")
    meta_idx = meta.set_index("id")
    y_test = meta_idx.loc[test_ids, "boneage"].values
    X_test = _hog.extract_features(test_ids, images_dir, meta)
    return X_train, y_train, X_test, y_test


def get_intensity(images_dir, csv_path, splits_dir, out_dir, train_ids, test_ids, meta):
    cache_path = out_dir / "features_cache_intensidade.joblib"
    if cache_path.exists():
        print("Reaproveitando cache de intensidade (treino) ja existente...")
        cache = joblib.load(cache_path)
        X_train, y_train = cache["X_train"], cache["y_train"]
    else:
        print("Extraindo intensidade do treino (sem cache)...")
        meta_idx = meta.set_index("id")
        y_train = meta_idx.loc[train_ids, "boneage"].values
        X_train = _inten.extract_intensity_features(train_ids, images_dir, meta)

    print("Extraindo intensidade do TESTE (600 imagens, execucao unica)...")
    meta_idx = meta.set_index("id")
    y_test = meta_idx.loc[test_ids, "boneage"].values
    X_test = _inten.extract_intensity_features(test_ids, images_dir, meta)
    return X_train, y_train, X_test, y_test


def load_texture_split(texture_dir: Path, split: str):
    tex = pd.read_csv(texture_dir / f"texture_{split}.csv", dtype={"id": str})
    meta = pd.read_csv(texture_dir / f"metadata_{split}.csv", dtype={"id": str})
    assert tex["id"].tolist() == meta["id"].tolist()
    X = tex.drop(columns="id").to_numpy(dtype=float)
    sex = meta["male"].to_numpy(dtype=float).reshape(-1, 1)
    X = np.hstack([X, sex])
    y = meta["boneage"].to_numpy(dtype=float)
    return X, y


def get_texture(texture_dir: Path):
    X_train, y_train = load_texture_split(texture_dir, "train")
    X_test, y_test = load_texture_split(texture_dir, "test")
    return X_train, y_train, X_test, y_test


def main(images_dir, csv_path, splits_dir, texture_dir, out_dir):
    images_dir = Path(images_dir)
    splits_dir = Path(splits_dir)
    texture_dir = Path(texture_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(csv_path, dtype={"id": str})
    train_ids = pd.read_csv(splits_dir / "train_ids.csv", dtype={"id": str})["id"].tolist()
    test_ids = pd.read_csv(splits_dir / "test_ids.csv", dtype={"id": str})["id"].tolist()
    print(f"Treino: {len(train_ids)} | Teste: {len(test_ids)} (avaliacao UNICA)")

    print("\n=== HOG ===")
    X_train, y_train, X_test, y_test = get_hog(images_dir, csv_path, splits_dir, out_dir, train_ids, test_ids, meta)
    hog_results = run_family("HOG", X_train, y_train, X_test, y_test)

    print("\n=== Intensidade ===")
    X_train, y_train, X_test, y_test = get_intensity(images_dir, csv_path, splits_dir, out_dir, train_ids, test_ids, meta)
    inten_results = run_family("Intensidade", X_train, y_train, X_test, y_test)

    print("\n=== Textura (LBP+GLCM) ===")
    X_train, y_train, X_test, y_test = get_texture(texture_dir)
    tex_results = run_family("Textura", X_train, y_train, X_test, y_test)

    familias = {"HOG": hog_results, "Textura (LBP+GLCM)": tex_results, "Intensidade": inten_results}
    modelos = ["media_treino", "SVR", "RandomForest", "GradientBoosting"]

    linhas = ["| Descritor | Modelo | MAE (meses) | RMSE (meses) | R² |", "|---|---|---|---|---|"]
    melhor = None
    for fam_nome, fam_res in familias.items():
        for mod_nome in modelos:
            m = fam_res[mod_nome]
            label = "Média do treino" if mod_nome == "media_treino" else mod_nome
            linhas.append(f"| {fam_nome} | {label} | {m['MAE_meses']:.2f} | {m['RMSE_meses']:.2f} | {m['R2']:.3f} |")
            if mod_nome != "media_treino" and (melhor is None or m["MAE_meses"] < melhor[2]):
                melhor = (fam_nome, mod_nome, m["MAE_meses"])

    tabela_md = "\n".join(linhas)
    resumo = f"\n\n**Melhor combinação no TESTE**: {melhor[0]} + {melhor[1]}, MAE = {melhor[2]:.2f} meses.\n"
    nota = (
        "\n\n> Avaliação única no conjunto de teste (600 imagens), com modelos cujos "
        "hiperparâmetros foram fixados exclusivamente a partir do treino/validação/CV. "
        "Nenhum ajuste foi feito após observar este resultado.\n"
    )

    (out_dir / "tabela_final_teste.md").write_text(tabela_md + resumo + nota, encoding="utf-8")
    with open(out_dir / "resultados_teste_todas_familias.json", "w") as f:
        json.dump(familias, f, indent=2, ensure_ascii=False)

    print("\n=== TABELA FINAL -- CONJUNTO DE TESTE (avaliação única) ===")
    print(tabela_md)
    print(resumo)
    print(f"\nSalvo em {out_dir / 'tabela_final_teste.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--texture-dir", required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.images_dir, args.csv, args.splits_dir, args.texture_dir, args.out_dir)
