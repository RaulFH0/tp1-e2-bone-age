"""Imagem de entrada comum aos descritores do TP1.

Use preprocess_image no HOG, na textura e na intensidade para que as três
famílias vejam os mesmos pixels. Não há recorte, remoção de fundo, realce
de contraste nem parâmetros ajustados com imagens de validação ou teste.
"""

from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Lê PNG, converte a cinza, redimensiona e devolve float32 em [0, 1]."""
    with Image.open(image_path) as source:
        gray = source.convert("L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8)
    return pixels.astype(np.float32) / 255.0
