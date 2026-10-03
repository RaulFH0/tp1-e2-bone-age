# Plano de fechamento do TP1 E2

Escopo autorizado: antecipar correções, avaliação reproduzível e redação para o prazo interno de 03/10. Base: main `ed2ade6`. Execução nesta sessão, sem publicação automática.

1. Testes de regressão: scaler ajustado no treino de cada dobra; IDs de textura iguais aos CSVs congelados; resolução alternativa mantém normalização; baseline por dobra e previsões com IDs.
2. Corrigir `03_tune_svr.py` com Pipeline e parâmetros prefixados; reforçar `07_compare_all_descriptors.py` com contrato dos IDs, colunas e valores.
3. Manter 224 como padrão; permitir 128 apenas no experimento de ablação. Implementar `08_cross_validation.py`: três famílias, três modelos, três dobras dentro dos 2.800 IDs de treino; estatísticas e previsões; comparação 224/128 de textura com o mesmo modelo. Grupos quando fornecidos; modo por imagem identificado como provisório e não conforme à independência por paciente.
4. Gerar script de instalação/execução no Colab, instruções, rascunho do artigo com lacunas explícitas de resultados e referências reais verificadas. Preservar validação e teste de seleção pela avaliação interna.
5. Rodar toda a suíte e execução sintética de ponta a ponta; nenhum número sintético vai ao artigo. Empacotar patch para Windows, sem sobrescrever IDs congelados, publicar arquivos de entrega para revisão.
