"""
01_split_data.py

Gera uma particao treino/validacao/teste PROVISORIA para o TP1 (RSNA Bone Age),
enquanto a particao oficial da equipe (responsabilidade do Carlos Daniel) nao
estiver congelada. Assim que a oficial existir, troque por ela e reaproveite
o restante do pipeline sem mudanca de codigo.

Premissas sobre o dataset (kaggle: kmader/rsna-bone-age):
- boneage-train-dataset.csv com colunas: id, boneage (meses), male (bool/0-1)
- cada linha = um exame/paciente unico (nao ha exames repetidos do mesmo paciente
  nesse dataset em particular) -> split por "id" already equivale a split por paciente.
  Se a equipe descobrir IDs de paciente repetidos na EDA do Carlos, ajuste aqui
  usando GroupShuffleSplit por paciente em vez de train_test_split simples.

Estratificacao: por faixa etaria (bins de 24 meses) + sexo, para manter a
distribuicao de idade e sexo semelhante nas 3 particoes.

Saida: train_ids.csv, val_ids.csv, test_ids.csv em data/splits/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
AGE_BIN_WIDTH_MONTHS = 24


def make_strata(df: pd.DataFrame) -> pd.Series:
    age_bin = (df["boneage"] // AGE_BIN_WIDTH_MONTHS).astype(int)
    sex = df["male"].astype(int)
    return age_bin.astype(str) + "_" + sex.astype(str)


def main(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    required_cols = {"id", "boneage", "male"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas esperadas ausentes no CSV: {missing}. "
            f"Confira o nome real das colunas do boneage-train-dataset.csv."
        )

    strata = make_strata(df)

    # Remove estratos com menos de 2 membros (train_test_split estratificado exige >=2)
    strata_counts = strata.value_counts()
    valid_mask = strata.isin(strata_counts[strata_counts >= 2].index)
    df_valid = df[valid_mask].reset_index(drop=True)
    strata_valid = strata[valid_mask].reset_index(drop=True)
    n_dropped = len(df) - len(df_valid)
    if n_dropped:
        print(f"Aviso: {n_dropped} exames em estratos raros ficaram de fora "
              f"da estratificacao (foram mantidos via split simples ao final).")

    train_df, temp_df, strata_train, strata_temp = train_test_split(
        df_valid, strata_valid,
        train_size=TRAIN_FRAC,
        random_state=RANDOM_SEED,
        stratify=strata_valid,
    )

    val_ratio_within_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_within_temp,
        random_state=RANDOM_SEED,
        stratify=strata_temp,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_df[["id"]].to_csv(out_path / "train_ids.csv", index=False)
    val_df[["id"]].to_csv(out_path / "val_ids.csv", index=False)
    test_df[["id"]].to_csv(out_path / "test_ids.csv", index=False)

    print(f"Split salvo em {out_path}/")
    print(f"  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    print(f"  semente fixa: {RANDOM_SEED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default="data/raw/boneage-train-dataset.csv",
        help="Caminho para o CSV com id, boneage, male",
    )
    parser.add_argument(
        "--out", default="data/splits",
        help="Pasta de saida para train_ids.csv / val_ids.csv / test_ids.csv",
    )
    args = parser.parse_args()
    main(args.csv, args.out)
