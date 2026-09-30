# TP1 · E2 · Baseline de idade óssea (equipe)

Repositório: https://github.com/RaulFH0/tp1-e2-bone-age

## Estrutura esperada

```
tp1_e2/
├── data/
│   ├── raw/
│   │   ├── boneage-train-dataset.csv
│   │   └── boneage-train-dataset/boneage-train-dataset/*.png
│   └── splits/          (IDs congelados — 4.000 imagens, semente 42, 2.800/600/600)
├── results/              (gerado pelos scripts 00, 02, 03, 04)
├── src/
│   ├── 00_audit_data.py
│   ├── 01_split_data.py         (split provisório — superado pelos IDs congelados)
│   ├── 02_hog_baseline.py       (Raul — HOG + SVR)
│   ├── 02_preprocess.py         (Carlos — textura: LBP + GLCM)
│   ├── 03_tune_svr.py           (Raul — ajuste de hiperparâmetro do SVR)
│   ├── 04_compare_models.py     (Raul — SVR × Random Forest × Gradient Boosting)
│   └── image_preprocessing.py   (Carlos — preprocess_image, comum a todos os descritores)
└── requirements.txt
```

Dados brutos (`data/raw/`) **não vão** para o repositório — só o caminho para obtê-los (ver seção "Origem dos dados").

## Origem dos dados

Dataset: RSNA Pediatric Bone Age Challenge (2017), disponibilizado no Kaggle como `kmader/rsna-bone-age`.

```bash
pip install kaggle
kaggle datasets download -d kmader/rsna-bone-age -p data/raw --unzip
```

Requer aceitar os termos do dataset na página do Kaggle antes do primeiro download, e um token de API (`~/.kaggle/kaggle.json`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
# 0. Carlos: auditar rótulos, imagens e possibilidade de agrupamento por paciente
python src/00_audit_data.py \
    --csv data/raw/boneage-train-dataset.csv \
    --images-dir data/raw/boneage-train-dataset/boneage-train-dataset \
    --out results/data_audit.json

# 1. Split provisório (histórico — a partição em uso é a congelada em data/splits/)
python src/01_split_data.py \
    --csv data/raw/boneage-train-dataset.csv \
    --out data/splits

# 2. HOG + SVR (Raul)
python src/02_hog_baseline.py \
    --images-dir data/raw/boneage-train-dataset/boneage-train-dataset \
    --csv data/raw/boneage-train-dataset.csv \
    --splits-dir data/splits \
    --out-dir results
```

Saída: `results/metrics_hog_svr.json` com MAE/RMSE/R² do SVR+HOG comparado à média do treino, no conjunto de validação.

## Uso via Google Colab

Se preferir rodar no Colab em vez de local, numa célula nova:

```python
!git clone https://github.com/RaulFH0/tp1-e2-bone-age.git
%cd tp1-e2-bone-age
!pip install -r requirements.txt
!kaggle datasets download -d kmader/rsna-bone-age -p data/raw --unzip
```

Depois é só rodar as mesmas linhas de `python src/00_...`, `src/01_...` e `src/02_...` com `!` na frente (`!python src/01_split_data.py ...`).

Se editar algo direto no Colab, baixe o notebook (File → Download → .ipynb) e suba de volta pro repositório via `git add` / `git commit` / `git push` — nunca deixe a única cópia só dentro do Colab.

## Observações importantes

- **Split provisório**: `01_split_data.py` faz um split estratificado por faixa etária + sexo com semente fixa, só para destravar o trabalho antes da partição oficial da equipe existir. A partição em uso agora é a congelada em `data/splits/` (ver seção abaixo) — o restante do pipeline não muda. O CSV padrão inclui `id`, `boneage` e `male`; `id` identifica uma imagem/exame, mas não prova que pacientes não se repetem. Verifique `results/data_audit.json`: se `patient_grouping.status` for `not_verifiable`, documente essa limitação e valide o protocolo com a equipe e o professor antes de chamar essa partição de "por paciente".
- **Auditoria de Carlos**: `00_audit_data.py` confere colunas, IDs duplicados, idades, sexo, correspondência `<id>.png`, uma amostra dos tamanhos das PNG e informa se há uma coluna explícita `patient_id`. O script não copia imagens nem dados brutos para o repositório. Para testar com dados pequenos, execute `python -m unittest discover -s tests -v`.
- **Nomes de arquivo de imagem**: o script assume `<id>.png`. Confira o formato real após o download.
- **Sem vazamento de dados**: o `StandardScaler` é ajustado (`fit`) somente no treino; a validação só usa `transform`. Mantenha esse padrão em todos os descritores e modelos.
- **Versões**: registre a versão exata de cada pacote (`pip freeze > requirements-lock.txt`) antes da entrega final, para reprodutibilidade.

## Amostra RSNA e extração de textura

Foi selecionada uma amostra de 4.000 das 12.611 imagens, com semente 42 e estratificação por faixa de idade óssea e sexo. Os IDs foram divididos em treino (2.800), validação (600) e teste (600), sem repetição de IDs de imagem.

O script `src/02_preprocess.py` lê os IDs congelados em `data/splits/`, converte cada PNG para escala de cinza, redimensiona para 224 × 224 pixels com Lanczos e calcula:

- **LBP uniforme**: 8 pontos, raio 1, histograma de 10 valores;
- **GLCM**: 16 níveis de cinza, distâncias 1 e 2, ângulos 0°, 45°, 90° e 135°; média e desvio de cinco propriedades;
- **Intensidade**: média dos pixels normalizados para o intervalo de 0 a 1.

São 21 características por imagem, identificadas pelo ID original. A execução no Colab gerou 2.800, 600 e 600 linhas; os 4.000 IDs foram conferidos com a amostra. Os CSVs de características e as imagens brutas não são enviados ao Git.

> **Limitação registrada**: a base utilizada não fornece uma chave verificável de paciente. Por isso, a ausência de IDs de imagem repetidos não comprova separação por paciente; o protocolo da divisão ainda depende de revisão metodológica pela equipe.

## Mesma imagem para HOG, textura e intensidade

`src/image_preprocessing.py` expõe `preprocess_image(caminho_png)`. Ela devolve uma matriz `float32` de 224 × 224 com pixels entre 0 e 1: conversão para cinza com Pillow (`L`), redimensionamento Lanczos e divisão por 255. Não aplica recorte, remoção de fundo ou ajuste de contraste. A textura em `src/02_preprocess.py` usa essa função antes de calcular LBP/GLCM.

Em um script dentro de `src/`, qualquer integrante pode usar exatamente a mesma imagem de entrada:

```python
from image_preprocessing import preprocess_image

image = preprocess_image(images_dir / f"{image_id}.png")
# Passe image à extração HOG ou intensidade.
```

Os IDs de 4.000 imagens e a semente 42 estão congelados para execução, mas a equipe ainda precisa revisar a limitação de agrupamento por paciente antes de chamar a partição de oficial.

---

## Modelagem — HOG e comparação de regressores (Raul)

Scripts nesta seção usam `preprocess_image` (mesma entrada que a textura) e os IDs congelados em `data/splits/`.

### `02_hog_baseline.py` — HOG + SVR (primeiro modelo)

```bash
python src/02_hog_baseline.py \
    --images-dir "data/raw/boneage-training-dataset/boneage-training-dataset" \
    --csv data/raw/boneage-training-dataset.csv \
    --splits-dir data/splits \
    --out-dir results
```

Saída: `results/metrics_hog_svr.json` (MAE/RMSE/R² do SVR+HOG vs. média do treino).

**Resultado atual (validação)**: média do treino MAE 33,71 · SVR MAE 27,88.

### `03_tune_svr.py` — ajuste de hiperparâmetro do SVR

Busca C/epsilon/gamma numa amostra do treino (`RandomizedSearchCV`, rápido), depois treina o modelo final com os melhores parâmetros no treino completo. Cacheia as features extraídas em `results/features_cache.joblib`.

```bash
python src/03_tune_svr.py \
    --images-dir "data/raw/boneage-training-dataset/boneage-training-dataset" \
    --csv data/raw/boneage-training-dataset.csv \
    --splits-dir data/splits \
    --out-dir results
```

Saída: `results/metrics_hog_svr_tuned.json`.

### `04_compare_models.py` — SVR × Random Forest × Gradient Boosting

Treina os três regressores clássicos exigidos na Semana 3 sobre o mesmo HOG e compara com a baseline da média do treino. Cacheia as features em `results/features_cache_hog_oficial.joblib` (cache próprio, separado do `03`).

```bash
python src/04_compare_models.py \
    --images-dir "data/raw/boneage-training-dataset/boneage-training-dataset" \
    --csv data/raw/boneage-training-dataset.csv \
    --splits-dir data/splits \
    --out-dir results
```

Saída: `results/metrics_hog_comparacao_modelos.json`.

**Resultado atual (validação)**:

| Modelo | MAE (meses) | RMSE (meses) | R² |
|---|---|---|---|
| Média do treino | 33,71 | 41,41 | ~0,00 |
| SVR | 27,88 | 34,82 | 0,29 |
| Random Forest | 26,85 | 33,96 | 0,33 |
| **Gradient Boosting** | **22,62** | **29,93** | **0,48** |

> **Importante**: sempre que `preprocess_image` ou os IDs congelados mudarem, apague os caches (`results/*.joblib`) antes de rodar de novo — features extraídas com uma versão antiga do preprocessamento não são comparáveis com os outros descritores da equipe.
