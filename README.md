# TP1 · E2 · Parte do Raul (literatura, HOG, treinamento)

Repositório: https://github.com/RaulFH0/-tp1-e2-bone-age

## Estrutura esperada
```
tp1_e2/
├── data/
│   ├── raw/
│   │   ├── boneage-train-dataset.csv
│   │   └── boneage-train-dataset/boneage-train-dataset/*.png
│   └── splits/          (gerado pelo script 01)
├── results/              (gerado pelo script 02)
├── src/
│   ├── 01_split_data.py
│   └── 02_hog_baseline.py
└── requirements.txt
```

Dados brutos (`data/raw/`) NÃO vão para o repositório — só o caminho para
obtê-los (ver seção "Origem dos dados").

## Origem dos dados
Dataset: RSNA Pediatric Bone Age Challenge (2017), disponibilizado no Kaggle
como `kmader/rsna-bone-age`.

```bash
pip install kaggle
kaggle datasets download -d kmader/rsna-bone-age -p data/raw --unzip
```

Requer aceitar os termos do dataset na página do Kaggle antes do primeiro
download, e um token de API (`~/.kaggle/kaggle.json`).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução
```bash
# 1. Split provisório treino/val/teste (semente fixa = 42)
python src/01_split_data.py \
    --csv data/raw/boneage-train-dataset.csv \
    --out data/splits

# 2. Extração HOG + treino/avaliação do SVR
python src/02_hog_baseline.py \
    --images-dir data/raw/boneage-train-dataset/boneage-train-dataset \
    --csv data/raw/boneage-train-dataset.csv \
    --splits-dir data/splits \
    --out-dir results
```

Saída: `results/metrics_hog_svr.json` com MAE/RMSE/R² do SVR+HOG comparado à
média do treino, no conjunto de validação.

## Uso via Google Colab
Se preferir rodar no Colab em vez de local, numa célula nova:
```python
!git clone https://github.com/RaulFH0/-tp1-e2-bone-age.git
%cd -tp1-e2-bone-age
!pip install -r requirements.txt
!kaggle datasets download -d kmader/rsna-bone-age -p data/raw --unzip
```
Depois é só rodar as mesmas duas linhas de `python src/01_...` e `python src/02_...`
com `!` na frente (`!python src/01_split_data.py ...`).

Se editar algo direto no Colab, baixe o notebook (File → Download → .ipynb) e
suba de volta pro repositório via `git add` / `git commit` / `git push` — nunca
deixe a única cópia só dentro do Colab.

## Observações importantes
- **Split provisório**: `01_split_data.py` faz um split estratificado por
  faixa etária + sexo com semente fixa, só para destravar o trabalho antes da
  partição oficial da equipe (responsabilidade do Carlos Daniel) existir.
  Assim que a oficial for definida, troque os CSVs em `data/splits/` — o
  restante do pipeline não muda.
- **Nomes de arquivo de imagem**: o script assume `<id>.png`. Confira o
  formato real após o download e ajuste `02_hog_baseline.py` se necessário.
- **Sem vazamento de dados**: o `StandardScaler` é ajustado (`fit`) somente
  no treino; a validação só usa `transform`. Mantenha esse padrão em todos os
  descritores e modelos futuros (textura, intensidade, RF, GB).
- **Versões**: registre a versão exata de cada pacote (`pip freeze >
  requirements-lock.txt`) antes da entrega final, para reprodutibilidade.
