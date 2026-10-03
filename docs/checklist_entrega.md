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
| Contribuição individual | docs/TP1_E2_contribuicao_para_preencher.pdf: formulário do Anexo A. Matrículas, contatos, atividades, matriz e proposta 60%/40% preenchidos. Validar em equipe, confirmar a composição oficial da E2 e colher as assinaturas dos integrantes oficiais; critério/evidências em docs/contribuicao_por_commits.md. |
| Protocolo por paciente/exame | Pendência científica da seção 4.4. CSV público não contém chave de paciente. Não criar patient_id a partir de id. Obter chave verificável ou esclarecer formalmente a alternativa com o professor. |
| Avaliação final | A execução atual preserva os 600 IDs de teste. Confirmar com a equipe se já foram avaliados. Consolidar essa avaliação antes de outro acesso ao teste; não substituir por números de validação ou CV. |
| Revisão dos autores | Ler artigo, conferir referências e declaração de apoio de IA; responsabilidade final dos signatários. |
| Submissão | Reunir PDF final, ZIP de fontes LaTeX e contribuição assinada em TP1_E2_RSNA.zip; informar URL pública do repositório no AVA. |

## Ordem para concluir

1. Integrar o pacote técnico pelo PR, incluindo README, auditoria, scripts, fontes SBC e formulário preenchido para revisão.
2. Consultar a reprodução completa já verificada e seus registros de métricas/rastreabilidade.
3. Confirmar protocolo e avaliação final com a equipe/professor.
4. Atualizar o artigo somente com informações e resultados confirmados; recompilar e conferir quatro páginas.
5. Revisar e assinar a contribuição. O formulário preenchido sem assinaturas não atende à entrega assinada.
6. Conferir o ZIP final e enviar no AVA. O pacote técnico de atualização não é o ZIP de submissão.

## Informação necessária dos autores

Atividades efetivas e percentuais acordados; resultado/arquivos de eventual teste já executado; resposta do professor sobre paciente/exame. Assinaturas manuscritas ou digitais são providenciadas pelos autores.
