"""
07_compare_all_descriptors.py

Comparacao final entre as tres familias de descritores da equipe: HOG
(Raul), textura LBP+GLCM (Carlos) e intensidade (Raul, contingencia pela
ausencia da Rilari). Fecha a "grade minima de resultados" da Semana 3.

A textura chega pronta (CSV) no pacote entregue pelo Carlos -- este script
so precisa treinar os mesmos 3 regressores sobre ela, igual ja foi feito
para HOG (04_compare_models.py) e intensidade (05_intensity_baseline.py).

Uso:
    python 07_compare_all_descriptors.py \
        --texture-dir "data/textura_carlos" \
        --out-dir results

Pressupõe que resultados HOG e intensidade ja foram gerados (scripts 04 e
05) em --out-dir, para montar a tabela final combinada.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_SEED = 42

MODELS = {
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=1.0),
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


def load_texture_split(texture_dir: Path, split: str):
    tex = pd.read_csv(texture_dir / f"texture_{split}.csv", dtype={"id": str})
    meta = pd.read_csv(texture_dir / f"metadata_{split}.csv", dtype={"id": str})
    assert tex["id"].tolist() == meta["id"].tolist(), (
        f"IDs de texture_{split}.csv e metadata_{split}.csv fora de ordem — "
        "não reordene os arquivos do pacote do Carlos."
    )
    X = tex.drop(columns="id").to_numpy(dtype=float)
    sex = meta["male"].to_numpy(dtype=float).reshape(-1, 1)
    X = np.hstack([X, sex])  # mesma convenção de HOG/intensidade: sexo concatenado
    y = meta["boneage"].to_numpy(dtype=float)
    return X, y


def run_texture(texture_dir: Path, out_dir: Path):
    X_train, y_train = load_texture_split(texture_dir, "train")
    X_val, y_val = load_texture_split(texture_dir, "val")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    baseline_pred = np.full_like(y_val, fill_value=y_train.mean(), dtype=float)
    results = {"media_treino": evaluate(y_val, baseline_pred)}

    for name, model in MODELS.items():
        print(f"Treinando {name} (textura)...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        results[name] = evaluate(y_val, y_pred)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_textura_comparacao_modelos.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Comparação de modelos (Textura LBP+GLCM, validação) ===")
    print(f"{'Modelo':<18} {'MAE (meses)':>12} {'RMSE (meses)':>14} {'R2':>8}")
    for name, m in results.items():
        print(f"{name:<18} {m['MAE_meses']:>12.2f} {m['RMSE_meses']:>14.2f} {m['R2']:>8.3f}")

    return results


def build_final_table(out_dir: Path, texture_results: dict):
    with open(out_dir / "metrics_hog_comparacao_modelos.json") as f:
        hog = json.load(f)
    with open(out_dir / "metrics_intensidade_comparacao_modelos.json") as f:
        intensidade = {k: v for k, v in json.load(f).items() if k != "_nota"}

    familias = {"HOG": hog, "Textura (LBP+GLCM)": texture_results, "Intensidade": intensidade}
    modelos = ["media_treino", "SVR", "RandomForest", "GradientBoosting"]

    linhas = []
    linhas.append("| Descritor | Modelo | MAE (meses) | RMSE (meses) | R² |")
    linhas.append("|---|---|---|---|---|")
    melhor = None
    for fam_nome, fam_res in familias.items():
        for mod_nome in modelos:
            m = fam_res[mod_nome]
            label = "Média do treino" if mod_nome == "media_treino" else mod_nome
            linhas.append(f"| {fam_nome} | {label} | {m['MAE_meses']:.2f} | {m['RMSE_meses']:.2f} | {m['R2']:.3f} |")
            if mod_nome != "media_treino" and (melhor is None or m["MAE_meses"] < melhor[2]):
                melhor = (fam_nome, mod_nome, m["MAE_meses"])

    tabela_md = "\n".join(linhas)
    resumo = f"\n\n**Melhor combinação geral**: {melhor[0]} + {melhor[1]}, MAE = {melhor[2]:.2f} meses.\n"

    out_path = out_dir / "tabela_comparativa_final.md"
    out_path.write_text(tabela_md + resumo, encoding="utf-8")

    with open(out_dir / "resultados_finais_todas_familias.json", "w") as f:
        json.dump(familias, f, indent=2, ensure_ascii=False)

    print("\n=== TABELA COMPARATIVA FINAL ===")
    print(tabela_md)
    print(resumo)
    print(f"\nSalvo em {out_path} e em resultados_finais_todas_familias.json")


def main(texture_dir: str, out_dir: str):
    texture_dir = Path(texture_dir)
    out_dir = Path(out_dir)

    for needed in ["metrics_hog_comparacao_modelos.json", "metrics_intensidade_comparacao_modelos.json"]:
        if not (out_dir / needed).exists():
            raise FileNotFoundError(
                f"{needed} não encontrado em {out_dir}/. Rode 04_compare_models.py e "
                "05_intensity_baseline.py antes deste script."
            )

    texture_results = run_texture(texture_dir, out_dir)
    build_final_table(out_dir, texture_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture-dir", required=True, help="Pasta com o zip do Carlos já extraído")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.texture_dir, args.out_dir)
