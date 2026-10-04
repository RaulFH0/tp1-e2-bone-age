# Fechamento experimental — fluxo de reprodução

A execução original de CV com imagens reais foi concluída. Seus resultados internos estão em docs/resultados_oof/; o pacote completo continua fora do Git. O README define um fluxo novo, sem depender de ZIPs de correção ou células antigas.

1. Em ambiente novo, instalar requirements-colab.txt e executar python -m unittest discover -s tests -v.
2. Reutilizar os quatro CSVs de data/splits/; não refazer a divisão durante a reprodução.
3. Executar python src/10_reproduce_from_raw.py --allow-image-folds. No Colab, utilizar o notebook 03 atualizado.
4. Salvar verificacao_limpa_TP1.zip, com auditoria, textura alinhada, CV, OOF, versões, parâmetros, hashes e figuras.
5. Conferir a diferença das métricas em verificacao_reproducibilidade.json. Diferenças relevantes exigem investigação, não alteração dos números do artigo.

O modo por imagem permanece provisório. Se houver id,patient_id verificável, usar --groups-csv e revisar os próprios splits antes de declarar independência por paciente. Não inventar grupos a partir do ID da imagem.

As OOF usam somente o treino. A seleção pela mesma CV não é avaliação externa independente. A avaliação final dos 600 IDs de teste foi publicada por Raul em e3b12d7 (04/10/2026) e incorporada ao artigo: HOG+GB, MAE 22,14 meses, RMSE 28,29 meses e R² 0,520. Os parâmetros foram fixados antes do teste, conforme o registro; não usar os resultados para novos ajustes. Esta atualização documental não reexecutou o teste.

O script 10 refaz a textura 224, extrai HOG/intensidade e roda as configurações fixas dos três modelos e a ablação de textura 128 com GB. O script 08 ajusta scaler e baseline no treino de cada dobra. O script 09 e a montagem de radiografias geram figuras sobre as OOF.

Testes automáticos usam dados pequenos para verificar contratos e rejeitar vazamento/IDs incorretos. Métricas de fixtures não são resultados experimentais e não entram no artigo. A reprodução real registra evidência separadamente.

Revisão de 04/10/2026: 21 testes aprovados. O script 11 passou a conferir splits congelados, schema de textura e rótulos originais, sem usar caches antigos sem proveniência. As métricas publicadas não foram alteradas, e nenhuma avaliação real de teste foi repetida. Para conferir a execução anterior, recuperar seus caches/manifesto/previsões existentes; a correção do código não comprova retroativamente a origem desses dados.
