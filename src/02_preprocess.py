"""Pré-processamento e descritores de textura da amostra RSNA."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

SIZE = (224, 224)
GLCM_PROPERTIES = (
    "contrast", "dissimilarity", "homogeneity", "energy", "correlation"
)


def extract_features(image_path):
    with Image.open(image_path) as source:
        image = source.convert("L").resize(SIZE, Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.uint8)

    normalized = pixels.astype(np.float32) / 255.0

    lbp = local_binary_pattern(pixels, P=8, R=1, method="uniform")
    counts = np.bincount(lbp.astype(np.int64).ravel(), minlength=10)[:10]
    features = {
        f"lbp_{index}": float(count / lbp.size)
        for index, count in enumerate(counts)
    }

    # A GLCM usa 16 níveis de cinza para manter a extração reproduzível.
    quantized = (pixels // 16).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )
    for name in GLCM_PROPERTIES:
        values = graycoprops(matrix, name)
        features[f"glcm_{name}_mean"] = float(values.mean())
        features[f"glcm_{name}_std"] = float(values.std())

    features["pixel_mean"] = float(normalized.mean())
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    splits = {}
    for name in ("train", "val", "test"):
        path = Path("data/splits") / f"{name}_ids.csv"
        ids = pd.read_csv(path, dtype={"id": str})["id"].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"IDs repetidos em {path}")
        splits[name] = ids

    all_ids = [image_id for ids in splits.values() for image_id in ids]
    if len(all_ids) != 4000 or len(set(all_ids)) != 4000:
        raise ValueError("A divisão deve ter 4.000 IDs únicos.")

    missing = [
        image_id for image_id in all_ids
        if not (args.images / f"{image_id}.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Imagens ausentes; exemplos: {missing[:5]}")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        rows = []
        for image_id in ids:
            rows.append({
                "id": image_id,
                **extract_features(args.images / f"{image_id}.png"),
            })
        destination = args.output / f"texture_{name}.csv"
        pd.DataFrame(rows).to_csv(destination, index=False)
        print(f"{name}: {len(rows)} imagens -> {destination}")


if __name__ == "__main__":
    main()
