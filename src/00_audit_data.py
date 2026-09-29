"""Audit the RSNA Bone Age training labels and PNG image correspondence.

Run before freezing train/validation/test partitions. This script deliberately
does not infer patient identity from an image ID.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


def audit_dataset(csv_path: Path, images_dir: Path) -> dict:
    if not csv_path.is_file():
        raise ValueError(f"CSV não encontrado: {csv_path}")
    if not images_dir.is_dir():
        raise ValueError(f"Pasta de imagens não encontrada: {images_dir}")

    labels = pd.read_csv(csv_path, dtype={"id": "string", "patient_id": "string"})
    required = {"id", "boneage", "male"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")
    if labels.empty:
        raise ValueError("CSV sem linhas de dados")
    if labels["id"].isna().any() or labels["id"].str.strip().eq("").any():
        raise ValueError("IDs de imagem vazios")
    if labels["id"].duplicated().any():
        raise ValueError("IDs de imagem duplicados no CSV")

    age = pd.to_numeric(labels["boneage"], errors="coerce")
    if age.isna().any() or (age < 0).any():
        raise ValueError("Idades ósseas ausentes ou inválidas")
    sex = labels["male"].astype("string").str.strip().str.lower().map(
        {"true": "male", "1": "male", "false": "female", "0": "female"}
    )
    if sex.isna().any():
        raise ValueError("Valores ausentes ou inválidos na coluna male")

    image_paths = [images_dir / f"{image_id}.png" for image_id in labels["id"]]
    missing_images = [path.stem for path in image_paths if not path.is_file()]
    found_paths = [path for path in image_paths if path.is_file()]
    sampled_dimensions = []
    for path in found_paths[:12]:
        with Image.open(path) as image:
            sampled_dimensions.append({
                "id": path.stem, "width": image.width, "height": image.height,
                "mode": image.mode,
            })

    patient_grouping = {
        "status": "not_verifiable",
        "note": "O id da imagem/exame não comprova identidade única do paciente; "
                "não declarar partição por paciente sem uma chave verificável.",
    }
    if "patient_id" in labels.columns:
        patient_id = labels["patient_id"].str.strip()
        if patient_id.isna().any() or patient_id.eq("").any():
            raise ValueError("patient_id contém valores ausentes")
        patient_grouping = {
            "status": "available", "column": "patient_id",
            "unique_groups": int(patient_id.nunique()),
        }

    age_bins = (age // 24).astype(int).value_counts().sort_index()
    return {
        "source_csv": csv_path.name,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "columns": labels.columns.tolist(),
        "n_rows": int(len(labels)),
        "age_months": {
            "minimum": float(age.min()), "median": float(age.median()),
            "maximum": float(age.max()),
        },
        "age_bins_24_months": {
            f"{bin_id * 24}-{bin_id * 24 + 23}": int(count)
            for bin_id, count in age_bins.items()
        },
        "sex_counts": {
            "female": int((sex == "female").sum()),
            "male": int((sex == "male").sum()),
        },
        "images": {
            "found": len(found_paths), "missing": len(missing_images),
            "missing_id_examples": missing_images[:10],
            "sampled_dimensions": sampled_dimensions,
        },
        "patient_grouping": patient_grouping,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/data_audit.json"))
    args = parser.parse_args()
    try:
        report = audit_dataset(args.csv, args.images_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Erro na auditoria: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Auditoria salva em {args.out}")
    print(f"Rótulos: {report['n_rows']} | PNG encontradas: "
          f"{report['images']['found']} | ausentes: {report['images']['missing']}")
    print(f"Agrupamento por paciente: {report['patient_grouping']['status']}")


if __name__ == "__main__":
    main()
