# Artigo TP1 E2: versão para revisão
PDF de quatro páginas e fonte LaTeX, com resultados reais de CV interna.
Esta versão não representa o encerramento da entrega do TP1.

## Abrir e editar
1. Importar este ZIP como novo projeto no Overleaf.
2. Selecionar main.tex como documento principal e pdfLaTeX como compilador.
3. Alterar texto ou identificação e recompilar; manter o limite de quatro páginas.
Alternativa local: executar pdflatex main.tex duas vezes.

## Identificação e pendências
- Afiliação informada pelos autores: Sistemas de Informação, UniCatólica – Centro Universitário Católica de Quixadá, Quixadá – CE, Brasil.
- Confirmar com o professor o protocolo de paciente/exame. IDs de imagem disjuntos não comprovam independência por paciente.
- Consolidar a avaliação externa já existente na equipe antes de novo teste. As métricas impressas são da CV no treino, não do conjunto de teste.
- Revisar artigo, referências e declaração de IA em equipe.
- Produzir a tabela individual do Anexo A com atividades efetivas, matrículas, percentuais acordados e assinaturas. Esse documento separado não está incluído.
- O ZIP de fontes é um dos componentes. A entrega final do AVA deve reunir artigo, fontes e contribuição assinada.

## Dados e rastreabilidade
Base: kmader/rsna-bone-age, versão 2; amostra de 4.000 IDs, semente 42.
Treino/validação/teste: 2.800/600/600 imagens.
CV: três dobras apenas no treino, com sexo como covariável.
Código e resultados: https://github.com/RaulFH0/tp1-e2-bone-age
Base conferida: main 5998f07 (PR #6 mesclado).
Pacote auditado: resultados_fechamento_TP1.zip.
SHA-256: 6c95a1d3c6f1ccaf9cfe07f9b62a8d763132e81a308c2390ebaee0855461c4c3.
OOF CSV: 5e1c67eb8f8a25e22a04b6252ce1c9d095287b0bab4c58a4a3e318333dfacd7c.

Figura 1: dois menores erros OOF (4220, 14986) e maior erro em cada faixa extrema (<72 meses: 3337; >=192 meses: 4199), dentre 2.800 previsões HOG+GB. Imagens reais com pré-processamento comum, sem realce adicional. casos_figura.csv registra as previsões; seleção ilustrativa.
Figura 2: Bland-Altman e MAE por idade das mesmas 2.800 previsões OOF.
Figuras complementares, inclusive dispersão e oito casos extremos, permanecem no repositório.
Os títulos das 11 referências contêm links para DOI/página da publicação.

## Formatação
sbc-template.sty preservado sem alteração, do espelho institucional UEFS em TEMPLATE_SOURCE.txt.
A4, coluna única, Times 12; margens laterais 3 cm, superior 3,5 cm, inferior 2,5 cm; legendas no estilo SBC.
Conferência: quatro páginas renderizadas, 11 referências, oito linhas em cada resumo.
verificacao_artigo.json registra controles e hashes.
