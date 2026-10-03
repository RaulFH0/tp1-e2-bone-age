"""Avaliação interna comum e ablação de resolução, sem usar val/test.

As dobras por imagem são provisórias: não comprovam independência por paciente.
Forneça --groups-csv (id,patient_id) quando houver chave verificável.
Sem essa chave, --allow-image-folds registra explicitamente a limitação.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skimage.feature import hog

from image_preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "src" / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEXTURE = load_script("02_preprocess")
COMPARISON = load_script("07_compare_all_descriptors")
HOG = load_script("02_hog_baseline")


def metrics(y, predicted):
    return {"MAE_meses": float(mean_absolute_error(y, predicted)),
            "RMSE_meses": float(np.sqrt(mean_squared_error(y, predicted))),
            "R2": float(r2_score(y, predicted))}


def evaluate_folds(X, y, ids, folds, models, family):
    """Cada scaler e baseline usa somente o treino da dobra; retorna OOF com IDs."""
    records = []
    scores = {name: [] for name in ["media_treino", *models]}
    if len(ids) != len(set(ids)) or len(X) != len(y) or len(y) != len(ids):
        raise ValueError("IDs duplicados ou tamanhos incompatíveis")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dados não finitos")
    covered = []
    for fold_id, (train, validation) in enumerate(folds, 1):
        if len(set(train) & set(validation)):
            raise ValueError("Treino e validação da dobra se sobrepõem")
        covered.extend(validation.tolist())
        predictions = {"media_treino": np.full(len(validation), y[train].mean())}
        for name, estimator in models.items():
            model = Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])
            print(f"{family}: dobra {fold_id}/{len(folds)}, {name}", flush=True)
            started = time.monotonic()
            model.fit(X[train], y[train])
            predictions[name] = model.predict(X[validation])
            print(f"{family}: dobra {fold_id}/{len(folds)}, {name} concluído em "
                  f"{time.monotonic() - started:.1f}s", flush=True)
        for name, predicted in predictions.items():
            scores[name].append(metrics(y[validation], predicted))
            for row, value in zip(validation, predicted):
                records.append({"id": ids[row], "family": family, "model": name,
                                "fold": fold_id, "boneage": float(y[row]),
                                "y_pred": float(value)})
    if sorted(covered) != list(range(len(ids))):
        raise ValueError("Cada ID deve ter exatamente uma previsão fora do treino")
    summary = {}
    for name, values in scores.items():
        summary[name] = {metric: {"mean": float(np.mean([v[metric] for v in values])),
                                 "std": float(np.std([v[metric] for v in values], ddof=1))}
                         for metric in ("MAE_meses", "RMSE_meses", "R2")}
        summary[name]["folds"] = values
    return summary, pd.DataFrame(records)


def frozen_ids(splits_dir):
    ids = {split: pd.read_csv(splits_dir / f"{split}_ids.csv", dtype={"id": str})["id"].tolist()
           for split in ("sample", "train", "val", "test")}
    for name, count in {"sample": 4000, "train": 2800, "val": 600, "test": 600}.items():
        if len(ids[name]) != count or len(set(ids[name])) != count:
            raise ValueError(f"Split {name} deve conter {count} IDs únicos")
    union = ids["train"] + ids["val"] + ids["test"]
    if len(set(union)) != 4000 or set(union) != set(ids["sample"]):
        raise ValueError("Sobreposição ou diferença em relação à amostra congelada")
    return ids


def make_folds(labels, n_folds, groups=None):
    age_bin = pd.cut(labels.boneage, [0, 72, 132, 192, 240], right=False, labels=False)
    if age_bin.isna().any():
        raise ValueError("Idade fora das faixas documentadas")
    strata = age_bin.astype(str) + "_" + labels.male.astype(int).astype(str)
    if strata.value_counts().min() < n_folds:
        raise ValueError("Estrato insuficiente para o número de dobras")
    splitter = (StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
                if groups is not None else
                StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42))
    folds = list(splitter.split(np.zeros(len(labels)), strata, groups))
    if groups is not None:
        for train, validation in folds:
            if set(groups[train]) & set(groups[validation]):
                raise ValueError("Paciente repetido entre partições da dobra")
    return folds


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    ids = frozen_ids(args.splits_dir)
    meta = pd.read_csv(args.csv, dtype={"id": str})
    if meta.id.duplicated().any():
        raise ValueError("IDs duplicados nos rótulos")
    if not set(ids["sample"]).issubset(set(meta.id)):
        raise ValueError("Rótulos ausentes para a amostra")
    aligned = meta.set_index("id").loc[ids["train"]].reset_index()
    if not np.isin(aligned.male.astype(float), [0, 1]).all():
        raise ValueError("Sexo deve ser binário")
    groups = None
    if args.groups_csv:
        patient = pd.read_csv(args.groups_csv, dtype={"id": str, "patient_id": str})
        if patient.id.duplicated().any() or patient.patient_id.isna().any():
            raise ValueError("Chaves de paciente ausentes ou IDs duplicados")
        grouped = patient.set_index("id").loc[ids["sample"], "patient_id"]
        partition_groups = [set(grouped.loc[ids[s]]) for s in ("train", "val", "test")]
        if any(partition_groups[i] & partition_groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Splits congelados compartilham pacientes; revisar protocolo")
        groups = grouped.loc[ids["train"]].to_numpy()
    elif not args.allow_image_folds:
        raise ValueError("Sem chave de paciente. Esclareça o protocolo; modo provisório: --allow-image-folds")
    folds = make_folds(aligned, args.folds, groups)
    # Confere todos os arquivos do pacote; não utiliza os rótulos de val/test para ajuste.
    for split in ("train", "val", "test"):
        COMPARISON.load_texture_split(args.texture_dir, split, args.splits_dir)
        packaged = pd.read_csv(args.texture_dir / f"metadata_{split}.csv", dtype={"id": str})
        truth = meta.set_index("id").loc[ids[split]]
        if not np.array_equal(packaged.boneage.to_numpy(), truth.boneage.to_numpy()) or not np.array_equal(
            packaged.male.astype(float).to_numpy(), truth.male.astype(float).to_numpy()
        ):
            raise ValueError("Rótulos do pacote diferentes da base")
    missing = [i for i in ids["train"] if not (args.images_dir / f"{i}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Imagens ausentes: {missing[:5]}")
    X_texture, y = COMPARISON.load_texture_split(args.texture_dir, "train", args.splits_dir)
    matrices = {"Textura": X_texture}
    hog_rows, intensity_rows, smaller_rows = [], [], []
    for position, image_id in enumerate(ids["train"]):
        path = args.images_dir / f"{image_id}.png"
        image = preprocess_image(path)
        sex = float(aligned.male.iloc[position])
        # Confere a matriz entregue com a mesma implementação usada na ablação.
        reference = TEXTURE.extract_features(image)
        expected = np.asarray([reference[c] for c in COMPARISON.TEXTURE_COLUMNS])
        if not np.allclose(expected, X_texture[position, :-1], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Textura do pacote diverge da imagem/implementação: ID {image_id}")
        hog_rows.append(np.r_[hog(image, **HOG.HOG_PARAMS), sex])
        flat = image.ravel()
        histogram = np.histogram(flat, bins=32, range=(0, 1), density=True)[0]
        intensity_rows.append(np.r_[histogram, flat.mean(), flat.std(),
                                    np.percentile(flat, [25, 50, 75]), sex])
        if not args.skip_ablation:
            features = TEXTURE.extract_features(preprocess_image(path, size=(128, 128)))
            smaller_rows.append([features[c] for c in COMPARISON.TEXTURE_COLUMNS] + [sex])
        if (position + 1) % 100 == 0:
            print(f"Extração: {position + 1}/{len(ids['train'])}", flush=True)
    matrices["HOG"] = np.asarray(hog_rows)
    matrices["Intensidade"] = np.asarray(intensity_rows)
    models = {"SVR": SVR(C=10, epsilon=1, kernel="rbf"),
              "RandomForest": RandomForestRegressor(n_estimators=args.n_estimators, random_state=42, n_jobs=-1),
              "GradientBoosting": GradientBoostingRegressor(n_estimators=args.n_estimators, max_depth=3,
                                                            learning_rate=0.05, random_state=42)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results, predictions = {}, []
    for name, matrix in matrices.items():
        results, predicted = evaluate_folds(matrix, y, ids["train"], folds, models, name)
        all_results[name] = results
        predictions.append(predicted)
    if not args.skip_ablation:
        all_results["Textura128"], predicted = evaluate_folds(
            np.asarray(smaller_rows), y, ids["train"], folds,
            {"GradientBoosting": models["GradientBoosting"]}, "Textura128")
        predictions.append(predicted)
    # Dados e previsões ficam associados aos IDs; caches antigos não são reutilizados.
    for name, matrix in matrices.items():
        np.savez_compressed(args.out_dir / f"features_cv_{name}.npz", X=matrix, y=y, ids=ids["train"])
    pd.concat(predictions, ignore_index=True).to_csv(args.out_dir / "predictions_oof.csv", index=False)
    assignment = pd.DataFrame({"id": ids["train"], "fold": 0})
    for index, (_, validation) in enumerate(folds, 1):
        assignment.loc[validation, "fold"] = index
    assignment.to_csv(args.out_dir / "fold_assignments.csv", index=False)
    manifest = {"seed": 42, "n_folds": args.folds, "n_train": 2800,
                "protocol": "grouped" if groups is not None else "image_only_not_patient_verified",
                "val_test_used_for_selection": False, "std_ddof": 1,
                "model_params": {name: estimator.get_params() for name, estimator in models.items()},
                "python": platform.python_version(),
                "source_hashes": {p.name: sha256(p) for p in (ROOT / "src").glob("*.py")},
                "split_hashes": {p.name: sha256(p) for p in args.splits_dir.glob("*_ids.csv")},
                "labels_sha256": sha256(args.csv), "texture_train_sha256": sha256(args.texture_dir / "texture_train.csv"),
                "results": all_results}
    if args.groups_csv:
        manifest["groups_sha256"] = sha256(args.groups_csv)
    (args.out_dir / "metrics_cross_validation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.out_dir / "requirements-run.txt").open("w") as versions:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=versions)
    table = ["| Família | Modelo | MAE (meses) | RMSE (meses) | R² |",
             "|---|---|---|---|---|"]
    for family, results in all_results.items():
        for model, values in results.items():
            cells = [f"{values[m]['mean']:.2f} ± {values[m]['std']:.2f}" for m in ("MAE_meses", "RMSE_meses", "R2")]
            table.append(f"| {family} | {model} | " + " | ".join(cells) + " |")
    (args.out_dir / "tabela_cross_validation.md").write_text("\n".join(table), encoding="utf-8")
    print("\n".join(table))
    print("Avaliação interna concluída. Teste preservado. Verifique o campo protocol antes do artigo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--out-dir", type=Path, default=Path("results/cv"))
    parser.add_argument("--folds", type=int, default=3, choices=range(2, 11))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--allow-image-folds", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    main(parser.parse_args())
