# Conferência dos resultados reais — TP1 E2

**Conferência do pacote concluída.** A avaliação científica é interna, por imagem, com três dobras no conjunto de treino. Não constitui resultado no teste final nem comprova independência por paciente.

## Arquivo e integridade

- Arquivo original: `resultados_fechamento_TP1.zip` (47.354.549 bytes).
- SHA-256: `6c95a1d3c6f1ccaf9cfe07f9b62a8d763132e81a308c2390ebaee0855461c4c3` — igual ao informado pelo Colab.
- Todos os 12 membros do ZIP passaram a verificação CRC. Arquivo original preservado.
- Código utilizado: base `ed2ade6f303fc383f3248c12414cac564f09217f` com o pacote de correções documentado no manifesto.
- Python da execução: 3.13.15. Bibliotecas e parâmetros constam nos arquivos originais.

## IDs, matrizes e dobras

Os hashes dos quatro arquivos de IDs coincidem com os arquivos congelados. Amostra: 4.000 IDs; treino: 2.800; validação externa: 600; teste: 600. Os três conjuntos são disjuntos por ID de imagem e sua união coincide com a amostra.

As matrizes contêm 2.800 linhas, exatamente na ordem de `train_ids.csv`: textura 20 características + sexo; HOG 6.084 + sexo; intensidade 37 + sexo. Características, rótulos e previsões são finitos, com sexo binário e rótulos/covariável coincidentes entre famílias.

As dobras foram reproduzidas com `StratifiedKFold`, estratos idade/sexo, semente 42: 934, 933 e 933 imagens. As 39.200 previsões correspondem a 14 combinações, cada uma com exatamente uma previsão fora da dobra por ID. Não há IDs externos de validação/teste nas previsões.

Cada baseline foi conferida contra a média dos rótulos do treino interno da sua dobra. MAE, RMSE e R² foram recalculados por dobra, assim como média e desvio amostral (`ddof=1`). A maior diferença absoluta em relação ao JSON foi 7.11e-15, compatível com arredondamento numérico. Os hashes do código correspondem à implementação preparada, que ajusta o scaler no treino de cada dobra.

Foram aprovadas 159 verificações de integridade, estrutura e cálculo. A conferência utiliza o conteúdo do pacote; não refez os treinamentos nem a leitura dos PNGs originais. A identidade dos rótulos com a base original tem o hash registrado e a conferência feita pelo script executado no Colab.

## Comparação interna

Média ± desvio amostral entre três dobras. Todas as representações incluem sexo.

| Representação | Modelo | MAE (meses) | RMSE (meses) | R² |
|---|---|---|---|---|
| Baseline comum | Média do treino | 33,83 ± 0,99 | 41,48 ± 0,66 | -0,00 ± 0,00 |
| Textura 224 | SVR | 25,71 ± 0,41 | 33,44 ± 0,19 | 0,35 ± 0,01 |
| Textura 224 | Random Forest | 24,84 ± 0,93 | 32,16 ± 1,10 | 0,40 ± 0,03 |
| Textura 224 | Gradient Boosting | 24,81 ± 0,98 | 32,01 ± 1,34 | 0,40 ± 0,04 |
| HOG | SVR | 28,15 ± 0,89 | 35,37 ± 0,67 | 0,27 ± 0,01 |
| HOG | Random Forest | 26,73 ± 0,99 | 33,25 ± 0,89 | 0,36 ± 0,02 |
| HOG | Gradient Boosting | 22,21 ± 0,85 | 28,35 ± 0,85 | 0,53 ± 0,02 |
| Intensidade | SVR | 28,35 ± 0,74 | 36,69 ± 0,64 | 0,22 ± 0,02 |
| Intensidade | Random Forest | 25,76 ± 1,18 | 33,13 ± 1,60 | 0,36 ± 0,05 |
| Intensidade | Gradient Boosting | 25,95 ± 1,04 | 33,52 ± 1,67 | 0,35 ± 0,05 |
| Textura 128 | Gradient Boosting | 25,54 ± 1,51 | 32,59 ± 1,51 | 0,38 ± 0,04 |

HOG + Gradient Boosting apresentou o menor MAE médio entre as nove configurações de desenvolvimento: **22,21 ± 0,85 meses**, RMSE **28,35 ± 0,85 meses**, R² **0,53 ± 0,02**. O MAE é 34,35% inferior ao da baseline de média (**33,83 ± 0,99 meses**). Essa classificação é interna à validação cruzada usada para comparar as configurações. Não é uma avaliação externa independente do modelo escolhido.

## Ablação de resolução

Textura + Gradient Boosting: 224×224 alcançou MAE 24,81 ± 0,98 meses; 128×128 alcançou 25,54 ± 1,51 meses. A redução de resolução aumentou o MAE médio em **0,73 mês**. Diferenças pareadas 128 − 224 por dobra: 0,01, 1,28, 0,91 mês. O aumento ocorreu nas três dobras, mas a primeira diferença foi pequena. Não foi realizado teste de significância e a conclusão se restringe a este descritor/modelo.

## Erros OOF do candidato HOG + Gradient Boosting

Estas análises foram calculadas das previsões OOF já existentes, sem treinamento adicional ou consulta ao teste. Diferença: predito menos referência.

- Viés médio: +0,22 mês.
- Desvio amostral das diferenças: 28,36 meses.
- Limites de concordância descritivos, média ± 1,96 DP: −55,38 e +55,81 meses.
- O viés global pequeno acompanha erros sistemáticos de sinais opostos por idade. A figura não demonstra concordância clínica e não substitui a avaliação final.

| Faixa (anos) | Imagens | MAE OOF agrupado (meses) |
|---|---:|---:|
| 0–5 | 296 | 43,53 |
| 6–10 | 924 | 19,29 |
| 11–15 | 1465 | 17,59 |
| 16–19 | 115 | 49,60 |

As faixas extremas têm menos observações e erros maiores. As diferenças médias são +43,49 meses em 0–5 anos, +15,20 em 6–10, −14,07 em 11–15 e −49,60 em 16–19. Há compressão das previsões em direção a idades intermediárias nesta avaliação. O padrão é descritivo e não estabelece uma causa.

Por sexo: feminino n=1.283, MAE OOF agrupado 22,38 meses; masculino n=1.517, MAE 22,06 meses. Esses são valores agrupados sobre os 2.800 casos, diferentes da média de métricas por dobra usada na tabela principal.

## Pendências para entregar o TP1

1. Esclarecer o requisito de agrupamento por paciente/exame: os arquivos utilizados não contêm chave verificável. O protocolo permanece `image_only_not_patient_verified`.
2. Consolidar com a equipe a configuração escolhida e os resultados de avaliação externa já existentes antes de realizar nova avaliação no teste. Este ZIP contém apenas a CV interna no treino e a ablação.
3. Conferir as Figuras 1 e 2 no documento final renderizado. As radiografias OOF reais já foram recuperadas e verificadas no pacote separado.
4. Revisar artigo, afiliação, referências e declaração de IA, adaptar ao template SBC, renderizar e conferir quatro páginas.
5. Completar e assinar as contribuições reais dos integrantes e montar o pacote final exigido.

**Não é necessário repetir esta execução para recuperar seus resultados.** As métricas e as matrizes estão íntegras no ZIP recebido.

## Atualização: radiografias OOF conferidas

Recebido `radiografias_analise_oof_TP1.zip`, SHA-256
`dcaf97f116bde5e728f36a5d49251d59b243cfb44d61557542bae89ee9446607`.
O CRC do pacote foi aprovado. A origem das previsões e do código de
pré-processamento coincide com os hashes do pacote de CV. A seleção dos
quatro menores e quatro maiores erros foi reproduzida sobre os 2.800 casos
HOG + Gradient Boosting; IDs, dobras, referências e previsões coincidem.
Os oito PNGs estão em cinza, 224×224, e todos os casos pertencem ao treino.
A conferência visual da montagem confirmou a correspondência dos títulos
e a legibilidade. Os hashes dos PNGs originais são registros da execução
Colab; os originais não foram reenviados para recalcular esses hashes.

Menores erros: 4220, 14986, 4869, 1574 (0,06–0,12 mês). Maiores erros:
3337, 8460, 4199, 5047 (97,47–103,40 meses). Os exemplos extremos são
ilustrativos e não constituem amostra aleatória. Nenhum teste externo foi
usado ou modelo reajustado para gerar as radiografias.

O script `09_oof_error_analysis.py` reproduziu os valores de Bland–Altman
e erro por idade a partir das previsões reais. Também gera a dispersão
referência × previsão. Treze testes do código preparado foram aprovados.
O pacote de commit foi aplicado em uma cópia isolada da main `ed2ade6`: os
29 arquivos previstos foram preparados, incluindo as figuras PNG, e o
`git diff --cached --check` passou. O aplicador PowerShell foi revisado;
não foi executado neste ambiente Linux.

Este é um checkpoint de desenvolvimento. Não é declaração de conclusão
do artigo, de conformidade do agrupamento por paciente ou de avaliação
final no teste.
