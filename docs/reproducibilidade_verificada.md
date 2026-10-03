# Reprodução completa em ambiente limpo

Verificada em 03/10/2026, em ambiente virtual criado sem pacotes do sistema, Python 3.12.14. A instalação usou requirements-colab.txt; o snapshot novo está em docs/requirements_verificacao.txt. A execução original do Colab usou Python 3.13.15 e tem seu próprio snapshot preservado em requirements-lock.txt.

Comando executado: python src/10_reproduce_from_raw.py --dataset-dir <raiz da versão 2 baixada> --out-dir results/reproducibilidade --allow-image-folds.

- 15 testes aprovados; fontes Python e código do notebook compilados.
- Download da base original, auditoria de 12.611 rótulos/imagens, nenhuma ausente.
- Textura dos 4.000 IDs refeita, sem reutilizar matrizes ou caches do pacote original.
- Três famílias e três modelos, três dobras, 300 estimadores; baseline e ablação de resolução concluídas.
- As três matrizes de treino reextraídas são numericamente idênticas às originais, incluindo rótulos e ordem de IDs (maior diferença absoluta: zero).
- 39.200 previsões OOF conferidas: cada combinação cobre exatamente os 2.800 IDs de treino, uma vez por ID. Validação externa e teste não usados para seleção.
- Todas as médias e desvios das métricas coincidiram com a referência até erro de ponto flutuante: maior diferença absoluta 7,11e-15.
- HOG + Gradient Boosting: MAE 22,21 ± 0,85 meses; baseline 33,83 ± 0,99 meses.
- Auditoria, figuras, versões, matrizes e previsões preservadas em verificacao_limpa_TP1.zip. O JSON versionado registra hashes do código efetivamente executado, splits e auditoria.
- Tempo da execução a partir da base disponível: aproximadamente 66 minutos. O download foi verificado separadamente antes do comando. O tempo depende da máquina.

O log completo desta execução é docs/log_reproducibilidade.txt. A aplicação do conteúdo do pacote também foi conferida em clone isolado, com os arquivos preparados no Git e diff sem erros. O script PowerShell deve ser executado no computador Windows do projeto; não foi executado neste ambiente Linux.

Esta confirmação fecha a reprodução técnica do protocolo utilizado. Não comprova independência por paciente/exame e não substitui avaliação final externa, afiliação acadêmica ou contribuição assinada. O protocolo continua image_only_not_patient_verified.
