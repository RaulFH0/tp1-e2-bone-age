# Esclarecimento do protocolo de paciente/exame

A seção 4.4 da atividade exige partição por paciente/exame. A base RSNA pública auditada contém apenas id, boneage e male, e o id identifica a imagem. Os arquivos usados não permitem verificar se exames diferentes pertencem à mesma pessoa.

A publicação do desafio descreve radiografias desidentificadas, mas a descrição consultada não fornece correspondência id/paciente para esta amostra: Halabi et al., Radiology 2019, DOI https://doi.org/10.1148/radiol.2018180736. A ausência de uma correspondência disponível não é evidência de pacientes únicos.

Solicitação a encaminhar pela equipe: qual chave verificável deve ser usada para o agrupamento, ou qual protocolo alternativo o professor aceita para esta base? Até a resposta, o artigo declara CV provisória por imagem, sem afirmar conformidade com a exigência por paciente. Se houver chave, o script 08 aceita --groups-csv e rejeita sobreposição entre partições; pode ser necessário redefinir splits e repetir experimentos.

Nenhuma resposta ou autorização do professor foi presumida.
