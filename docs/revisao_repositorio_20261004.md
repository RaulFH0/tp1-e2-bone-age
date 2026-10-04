# Revisão geral do repositório — 04/10/2026

Base publicada: main `0cc180e4d089ea1703b48ed545c9506826b36e2c`. Referência de requisitos: TP1-baseline-rsna-turma788, seções 4.4, 6, 7, 9 e Anexo A. A revisão não repetiu CV, treinamento real ou avaliação real do teste.

## Conferido

| Área | Evidência e resultado |
|---|---|
| Amostra e partições | 4.000 IDs únicos; 2.800 treino, 600 validação, 600 teste; partições disjuntas e união igual à amostra; semente 42. |
| Auditoria | 12.611 imagens e rótulos, zero imagens ausentes no registro auditado. |
| Pré-processamento | Comum às três famílias: cinza, Lanczos 224×224, float32/255; parâmetros documentados. |
| Comparação | Três famílias e três regressores clássicos; baseline da média do treino; scaler e baseline ajustados dentro de cada dobra. |
| Reprodução | Fluxo README → script 10 → script 08 consistente; dependências diretas fixadas e reprodução limpa anterior documentada. |
| OOF e ablação | Resultados reais versionados e figuras identificadas como OOF do treino; ablação 224/128. |
| Artigo preparado | Quatro páginas, três integrantes, onze referências, CV e teste apresentados separadamente. PDF/LaTeX correspondentes. |
| Contribuição | Documento preparado com três integrantes e 50%/50%/0%; recebido com assinaturas de Carlos Daniel e Raul. |
| Testes desta revisão | 21 testes aprovados; seis novos casos sintéticos de validação da avaliação final. Sem dados reais do teste. |

## Correções preparadas

A main ainda apresenta artigo com dois autores, contribuição anterior e textos de teste pendente. Este pacote atualiza artigo, identificação, contribuição e documentação, preservando o nome `docs/artigo_sbc/TP1_E2_artigo_SBC.pdf` adotado por Raul.

O script 11 aceitava caches de HOG/intensidade sem IDs/proveniência e CSVs de textura sem comparar contra os splits congelados ou conferir o conjunto exato de atributos. Fixtures sintéticas demonstraram aceitação de rótulo 999 onde a anotação original era 12 e de ID/esquema inválido.

A correção extrai HOG/intensidade dos pixels comuns, reutiliza a validação dos 20 atributos e IDs do script 07, confere cardinalidade/disjunção pelo script 08 e compara idade/sexo com o CSV original antes de ajustar modelos. O resumo aponta o candidato selecionado pela CV; não escolhe o modelo pelo teste. Os modelos, parâmetros e métricas publicadas foram preservados.

## Pendências que esta revisão não resolve por suposição

1. **Agrupamento por paciente/exame.** A seção 4.4 exige essa partição. O CSV público usado contém ID de imagem e não uma chave verificável de paciente. O protocolo por imagem permanece provisório; obter chave ou esclarecimento do professor. A ausência de chave não demonstra que houve vazamento, mas impede comprovar independência por paciente.
2. **Proveniência do teste publicado.** As métricas agregadas de Raul em e3b12d7 estão preservadas. Os caches, matrizes, hashes e previsões individuais daquela execução não foram fornecidos para auditoria. Recuperar evidências existentes; corrigir o código não certifica retroativamente os resultados. Não retunar ou repetir o teste por causa desta revisão.
3. **Assinatura de Rílari.** O PDF recebido tem duas assinaturas digitais e percentuais 50%/50%/0%. O Anexo A exige assinatura de todos. Acrescentar a assinatura ao mesmo PDF sem alterar/reimprimir o conteúdo já assinado.
4. **Integração e envio.** As correções estão preparadas neste pacote; ainda precisam de commit, push e PR/merge. Conferir a versão final do artigo em equipe e enviar o ZIP do AVA com o link do repositório.

A inclusão dos três integrantes segue a orientação do professor relatada por Carlos Daniel. Essa confirmação de integrantes não equivale a aprovação do protocolo por imagem.
