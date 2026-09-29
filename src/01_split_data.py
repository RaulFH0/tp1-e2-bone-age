"""Partição provisória e reprodutível da amostra RSNA Bone Age.

Usa somente os IDs listados em sample_ids.csv, com 70% treino, 15%
validação e 15% teste, estratificando por quatro faixas de idade óssea
e sexo (semente 42). O campo id identifica imagem/exame, não paciente:
esta partição NÃO comprova separação por paciente e precisa de revisão
metodológica da equipe antes de ser tratada como definitiva.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
AGE_BINS = [0, 72, 132, 192, 240]
AGE_LABELS = ["0-5", "6-10", "11-15", "16-19"]


def split_sample(csv_path: str, sample_ids_path: str, out_dir: str) -> None:
    labels = pd.read_csv(csv_path)
    sample_ids = pd.read_csv(sample_ids_path)

    required = {"id", "boneage", "male"}
    if missing := required - set(labels.columns):
        raise ValueError(f"Colunas ausentes no CSV de rótulos: {sorted(missing)}")
    if list(sample_ids.columns) != ["id"]:
        raise ValueError("sample_ids.csv deve conter apenas a coluna id")
    if labels["id"].isna().any() or labels["id"].duplicated().any():
        raise ValueError("IDs inválidos ou duplicados no CSV de rótulos")
    if sample_ids.empty or sample_ids["id"].isna().any() or sample_ids["id"].duplicated().any():
        raise ValueError("Amostra vazia ou com IDs inválidos/duplicados")

    missing_ids = sorted(set(sample_ids["id"]) - set(labels["id"]))
    if missing_ids:
        raise ValueError(f"IDs da amostra ausentes no CSV de rótulos: {missing_ids[:5]}")

    data = (
        sample_ids.merge(labels, on="id", validate="one_to_one")
                  .sort_values("id")
                  .reset_index(drop=True)
    )
    data["faixa"] = pd.cut(
        data["boneage"], bins=AGE_BINS, right=False, labels=AGE_LABELS
    )
    if data[["faixa", "male"]].isna().any().any():
        raise ValueError("Amostra contém idade óssea ou sexo ausente/inválido")
    data["estrato"] = data["faixa"].astype(str) + "_" + data["male"].astype(str)

    train, rest = train_test_split(
        data, test_size=0.30, random_state=RANDOM_SEED,
        stratify=data["estrato"],
    )
    val, test = train_test_split(
        rest, test_size=0.50, random_state=RANDOM_SEED,
        stratify=rest["estrato"],
    )

    parts = {"train": train, "val": val, "test": test}
    sets = [set(part["id"]) for part in parts.values()]
    if set.union(*sets) != set(sample_ids["id"]) or sum(map(len, sets)) != len(data):
        raise AssertionError("Partição contém IDs repetidos ou não cobre a amostra")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        target = output / f"{name}_ids.csv"
        part[["id"]].sort_values("id").to_csv(target, index=False)
        print(f"{name}: {len(part)} IDs -> {target}")
    print(f"Semente: {RANDOM_SEED}; separação por paciente: não verificável")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/boneage-train-dataset.csv")
    parser.add_argument("--sample-ids", default="data/splits/sample_ids.csv")
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()
    split_sample(args.csv, args.sample_ids, args.out)
