# Fechamento experimental — execução linear

1. Instale `requirements.txt` e rode `python -m unittest discover -s tests -v`.
2. Reutilize `data/splits/`; não execute o script 01 para refazer a divisão.
3. Extraia `matrizes_textura_RSNA_4000.zip` em uma pasta separada. Localize a pasta que contém `texture_train.csv` e `metadata_train.csv`.
4. Use as imagens e o CSV de rótulos obtidos pela versão 2 de `kmader/rsna-bone-age`. O notebook novo resolve o caminho retornado pelo KaggleHub, sem assumir que `/kaggle/input` existe.
5. Rode o script 08. Ele extrai HOG e intensidade novamente sobre os IDs de treino para não confiar em caches antigos sem IDs. A textura 224 vem do pacote validado. A textura 128 é extraída apenas para a ablação; a versão padrão não é alterada.

```bash
python src/08_cross_validation.py \
  --images-dir PASTA_DAS_IMAGENS \
  --csv CAMINHO_DOS_ROTULOS.csv \
  --texture-dir PASTA_DO_PACOTE_TEXTURA \
  --out-dir results/cv \
  --allow-image-folds
```

`--allow-image-folds` é modo **provisório**, com independência por paciente não verificável. Não é aprovação do professor nem prova de atendimento à partição por paciente/exame. Se houver chave verificável, use `--groups-csv grupos.csv` contendo `id,patient_id`; os grupos precisam respeitar também os splits congelados. Não invente chaves de paciente a partir dos IDs das imagens.

O script usa três dobras estratificadas por idade e sexo dentro dos 2.800 IDs de treino. Validação e teste não são usados no ajuste nem na seleção dessas variantes. Não usa parâmetros previamente ajustados: a comparação mantém as configurações fixas dos três modelos, com scaler ajustado no treino de cada dobra. A busca de SVR no script 03 é separada e agora usa Pipeline. Não apresentar o resultado da busca como avaliação cruzada independente do modelo ajustado.

Saídas reais necessárias:

- `metrics_cross_validation.json`: média, desvio amostral (ddof=1), métricas por dobra, parâmetros e hashes;
- `tabela_cross_validation.md`: três famílias × três modelos e baseline, mais textura 128;
- `predictions_oof.csv`: IDs e previsões fora do treino de cada dobra;
- `fold_assignments.csv`: mesma atribuição das dobras para todas as variantes;
- `features_cv_*.npz`: matrizes de treino com IDs, idade e sexo incluído nas features;
- `requirements-run.txt`: versões efetivamente utilizadas.

A ablação compara textura 224 e 128 com Gradient Boosting, mantendo modelo, sexo e dobras. Não usar teste para escolher a resolução. Redimensionar altera a escala espacial dos descritores: o resultado deve ser interpretado como efeito do pipeline de resolução, e não como mudança isolada de um único osso.

O script 06 ainda analisa HOG + Gradient Boosting especificamente. Só apresente suas figuras como análise do melhor modelo se a comparação completa confirmar essa escolha. Após escolha e congelamento do protocolo, preparar a avaliação final do teste e seus gráficos sem retornar ao ajuste de parâmetros. A ausência de comprovação por paciente permanece uma pendência metodológica a esclarecer.

## Situação da verificação deste pacote

Onze testes aprovados. Fluxo completo do script 08 executado com radiografias sintéticas, os 4.000 IDs publicados, 2.800 IDs usados na CV, três famílias, três modelos, três dobras e ablação 224/128. Nessa verificação, as árvores foram reduzidas a dois estimadores para limitar custo; a configuração de uso real é 300 estimadores. Nenhuma métrica sintética é resultado clínico ou vai ao artigo. O script real ainda precisa ser executado no Colab da equipe.
