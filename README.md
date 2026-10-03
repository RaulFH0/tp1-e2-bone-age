# TP1 E2 — Idade óssea RSNA

Baseline de regressão com textura LBP/GLCM, HOG e intensidade. Autores do artigo: Carlos Daniel Reis da Silva e Raul Ferreira Holanda.

## Protocolo e números reportados

- Base: kmader/rsna-bone-age/versions/2, 12.611 radiografias de treino público.
- Amostra: 4.000 imagens, semente 42, estratificação por sexo e idade em [0,72), [72,132), [132,192) e [192,240) meses.
- IDs congelados em data/splits/: treino 2.800, validação 600, teste 600. Os arquivos são a referência de ordem e inclusão; não sobrescrevê-los durante a reprodução.
- Comparação do artigo: três dobras estratificadas, semente 42, apenas nos 2.800 IDs de treino. Scaler e baseline da média ajustados no treino de cada dobra.
- HOG + Gradient Boosting com sexo: MAE interno de **22,21 ± 0,85 meses**; média do treino: **33,83 ± 0,99 meses**. Média e desvio amostral entre dobras; não são métricas do teste externo.
- Sem chave verificável de paciente. O modo --allow-image-folds registra um protocolo provisório por imagem; não comprova independência por paciente nem autorização do professor. Uma chave id,patient_id permite --groups-csv, com verificação de grupos entre partições.

## Reprodução completa em sessão limpa do Colab

Após integrar os arquivos desta versão na main, abrir notebooks/03_fechamento_cv_ablacao.ipynb no Google Colab e executar a única célula de código em uma sessão nova. Ela:

1. Clona uma cópia isolada da main e registra o commit.
2. Instala requirements-colab.txt e executa os testes.
3. Obtém a versão 2 da base pelo KaggleHub e localiza os arquivos sem assumir /kaggle/input nem um ID específico de imagem.
4. Gera data_audit.json sobre os rótulos e as imagens reais.
5. Refaz a textura dos 4.000 IDs congelados, com 20 características e metadados alinhados. Não exige ZIP de textura nem ZIP de correção externo.
6. Executa CV de três famílias × três modelos, baseline e ablação de textura 224/128. Os 600 IDs de validação e os 600 de teste não entram no treinamento ou seleção dessa execução.
7. Gera tabelas, previsões OOF, Bland–Altman, dispersão, erro por faixa e radiografias ilustrativas; compara as métricas novas com a referência versionada.
8. Disponibiliza verificacao_limpa_TP1.zip. Salvar o ZIP e uma cópia do notebook executado.

O download tem aproximadamente 9,3 GiB. A CV pode demorar horas em CPU. O log informa a extração, o início e a conclusão de cada modelo/dobra. As porcentagens gerais indicam etapas concluídas; não estimam o tempo restante. O processamento ocorre na máquina do Colab quando o notebook é usado lá.

O KaggleHub pode exigir autenticação/aceitação de termos para acessar a base. Nesse caso, concluir a autenticação no ambiente de execução e repetir o comando; credenciais ficam fora do Git. Não trocar silenciosamente de versão ou conjunto de dados.

## Execução local equivalente

Python 3.12 ou 3.13. A execução original usou Python 3.13.15; as versões diretas do projeto estão fixadas.

    git clone https://github.com/RaulFH0/tp1-e2-bone-age.git
    cd tp1-e2-bone-age
    python -m venv .venv

Ativar o ambiente no Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

No Linux/macOS:

    source .venv/bin/activate

Instalar e verificar:

    python -m pip install -r requirements-colab.txt
    python -m unittest discover -s tests -v
    python src/10_reproduce_from_raw.py --allow-image-folds

Se a versão 2 já estiver extraída, informar a pasta que contém o CSV e as subpastas das imagens:

    python src/10_reproduce_from_raw.py --dataset-dir data/raw/rsna-bone-age-v2 --allow-image-folds

O nome do CSV do Kaggle é boneage-training-dataset.csv; a pasta pode ter dois níveis boneage-training-dataset. O script descobre a pasta por correspondência com os IDs dos rótulos. Não mover, copiar ou descompactar imagens dentro de data/splits/.

Para executar somente a auditoria, sem extração nem treinamento:

    python src/10_reproduce_from_raw.py --audit-only

Resultados ficam em results/reproducibilidade/; o ZIP fica em results/verificacao_limpa_TP1.zip. Dados brutos, caches e resultados completos permanecem fora do Git.

## Recuperar e conferir resultados existentes

A execução original já foi concluída e auditada. Não é preciso repeti-la apenas para consultar números ou gerar novamente gráficos. Extrair o pacote salvo resultados_fechamento_TP1.zip em results/cv/ e executar:

    python src/09_oof_error_analysis.py --predictions results/cv/predictions_oof.csv --out-dir results/analise_oof

Esse comando usa previsões existentes e não treina modelos. SHA-256 do pacote original: 6c95a1d3c6f1ccaf9cfe07f9b62a8d763132e81a308c2390ebaee0855461c4c3.

Referências versionadas: docs/resultados_oof/ e docs/figuras/. A verificação completa a partir das imagens é o fluxo do script 10; regenerar gráficos a partir de OOF é uma verificação parcial distinta.

## Pixels, descritores e modelos

src/image_preprocessing.py: cinza Pillow L, Lanczos 224×224, float32/255; sem recorte, remoção de fundo ou realce. A ablação muda a resolução para 128×128 no pipeline de textura.

| Família | Parâmetros | Atributos antes do sexo |
|---|---|---:|
| Textura | LBP uniforme P=8/R=1: 10 bins. GLCM: 16 níveis, distâncias 1/2, ângulos 0/45/90/135°, simétrica e normalizada; média e DP populacional de contraste, dissimilaridade, homogeneidade, energia e correlação | 20 |
| HOG | 9 orientações; células 16×16; blocos 2×2; L2-Hys | 6.084 |
| Intensidade | Histograma 32 bins em [0,1], densidade; média, DP e percentis 25/50/75 | 37 |

Sexo é acrescentado a todas as famílias. O campo pixel_mean do extrator antigo é excluído da comparação LBP/GLCM. CSVs de textura têm id e exatamente os 20 atributos documentados, na ordem dos splits.

Modelos fixos da CV: SVR RBF C=10/epsilon=1/gamma=scale; Random Forest 300 árvores/profundidade ilimitada/max_features=1.0/semente 42; Gradient Boosting 300 estimadores/profundidade 3/taxa 0,05/semente 42. Não confundir com a busca separada do SVR no script 03.

## Arquivos e versões

- src/00_audit_data.py: auditoria da base e indicação de chave explícita de paciente.
- src/01_split_data.py: utilitário para reproduzir a divisão a partir de sample_ids.csv. Não integra o fluxo principal, que lê os IDs publicados.
- src/02_preprocess.py: extrator original; inclui pixel_mean. O script 10 prepara somente os 20 atributos de textura exigidos pela comparação.
- Scripts 02 HOG e 03–07: desenvolvimento na validação única; não substituem a tabela por dobras.
- src/08_cross_validation.py: comparação interna e ablação.
- src/09_oof_error_analysis.py: gráficos e análise das OOF existentes.
- src/10_reproduce_from_raw.py: auditoria, textura, CV e figuras a partir da base original.
- requirements.txt: dependências diretas do cálculo, fixadas.
- requirements-colab.txt: instalação do cálculo e download pelo KaggleHub.
- requirements-lock.txt: pip freeze original completo do Colab, preservado como evidência. Não é instalador portátil: inclui pacotes e caminhos internos preinstalados do Colab.
- docs/versoes_execucao_cv.md: origem do registro de versões.

## Verificação realizada

Reprodução completa em ambiente virtual vazio: 15 testes aprovados, matrizes reextraídas numericamente idênticas e maior diferença entre as métricas novas e originais de 7,11e-15. Evidências e limites em docs/reproducibilidade_verificada.md/json.

## Artigo e fechamento

docs/artigo_rascunho.md conserva o texto de revisão. O PDF de quatro páginas e as fontes SBC ficam em docs/artigo_sbc/ nesta atualização. Não apresentar a CV interna como avaliação final independente nem como validação clínica.

A submissão final depende de esclarecer agrupamento por paciente/exame, consolidar a avaliação externa existente com a equipe e preencher/assinar a contribuição do Anexo A. O roteiro e os modelos de fechamento registram as pendências sem inventar informações ou assinaturas.
