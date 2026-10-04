# Fechamento TP1 E2

Referência: atividade TP1-baseline-rsna-turma788, seções 4.4, 6, 7 e Anexo A. Prazo oficial: 04/10/2026, 23h59. Esta lista distingue execução comprovada de informação ainda não fornecida.

| Item | Evidência / próxima ação |
|---|---|
| IDs, amostra e semente | data/splits/*.csv; 4.000 IDs, 2.800/600/600; semente 42. Reexecução conferiu inclusão e ordem. |
| Auditoria da base | docs/data_audit.json: 12.611 rótulos, 12.611 imagens localizadas, zero ausentes. |
| Pixels comuns às famílias | src/image_preprocessing.py; parâmetros no README. |
| Textura alinhada | CSVs já entregues no pacote matrizes_textura_RSNA_4000.zip; script 10 os refaz dos pixels originais. |
| CV e ablação reais | docs/resultados_oof/, execução original auditada; números internos no treino. |
| Ambiente original | requirements-lock.txt preserva o pip freeze completo; requirements-colab.txt instala as dependências diretas portáveis. |
| Reprodução limpa completa | Concluída em ambiente virtual vazio, sem reutilizar matrizes prontas. Matrizes idênticas e maior diferença entre métricas 7,11e-15; docs/reproducibilidade_verificada.md/json. |
| Artigo SBC | docs/artigo_sbc/: PDF de quatro páginas, LaTeX, duas figuras e onze referências. Afiliação preenchida: Sistemas de Informação, UniCatólica, Quixadá – CE, Brasil. |
| Contribuição individual | docs/TP1_E2_contribuicao_para_preencher.pdf: formulário do Anexo A. Matrículas, contatos, atividades, matriz e percentuais 50%/50%/0% preenchidos. O PDF recebido para o AVA contém assinaturas de Carlos Daniel e Raul; falta a assinatura de Rílari. O formulário versionado permanece sem assinaturas. Incluir Rílari com 0% de contribuição, conforme informado por Daniel; matrícula 2023010247 obtida do identificador do e-mail institucional informado por Daniel; critério/evidências em docs/contribuicao_por_commits.md. |
| Protocolo por paciente/exame | Pendência científica da seção 4.4. CSV público não contém chave de paciente. Não criar patient_id a partir de id. Obter chave verificável ou esclarecer formalmente a alternativa com o professor. |
| Avaliação final | Publicada por Raul em e3b12d7: results/resultados_teste_todas_familias.json e results/tabela_final_teste.md. Teste com 600 casos; HOG+GB: MAE 22,14, RMSE 28,29 meses, R² 0,520. Incorporada ao artigo; não reexecutada nesta atualização. Proveniência dos caches e previsões individuais daquela execução ainda não auditada. Script 11 corrigido com verificações de IDs, schema e rótulos para novas execuções. |
| Composição da equipe | Daniel informou em 04/10/2026 que o professor orientou incluir todos os integrantes: Carlos Daniel Reis da Silva, Raul Ferreira Holanda e Rílari Maia Castelo, inclusive sem contribuição. Item atualizado conforme essa orientação. |
| Revisão dos autores | Ler artigo, conferir referências e declaração de apoio de IA; responsabilidade final dos signatários. |
| Submissão | Reunir PDF final, ZIP de fontes LaTeX e contribuição assinada em TP1_E2_RSNA.zip; informar URL pública do repositório no AVA. |

## Ordem para concluir

1. Integrar o pacote desta revisão pelo PR: artigo com três integrantes, contribuição 50%/50%/0%, documentação e validações do script 11. O fluxo técnico de CV já está publicado na main.
2. Consultar a reprodução completa já verificada e seus registros de métricas/rastreabilidade.
3. Esclarecer o protocolo por paciente/exame e revisar em equipe os resultados de teste já incorporados.
4. Atualizar o artigo somente com informações e resultados confirmados; recompilar e conferir quatro páginas.
5. Obter a assinatura de Rílari no mesmo PDF já assinado por Carlos Daniel e Raul. Preservar o arquivo recebido; não reimprimir ou editar suas páginas.
6. Conferir o ZIP final e enviar no AVA. O pacote técnico de atualização não é o ZIP de submissão.

## Informação necessária dos autores

Resposta do professor sobre paciente/exame; evidências existentes da execução final de teste; assinatura de Rílari. As contribuições e percentuais estão registrados no documento recebido, assinado por Carlos Daniel e Raul.
