"""Casos do autor da peça em foco (CHAT_FOCUSED_ARTIFACT_AUTHOR).

O visitante tem uma peça à frente no minichat e a mensagem não diz o nome do
autor. OBRAS_DO_AUTOR devem receber «A» (o pedido passa a «obras de <autor>»);
SOBRE_A_PECA e OUTRA_COISA não devem receber «A» (seguem como hoje). Cada
mensagem é medida com as peças de PECAS, sem pedido anterior e com o pedido que
podia ter posto a peça à frente do visitante (ANTES_DA_PECA).

SEGUIMENTOS_AFINAR e SEGUIMENTOS_VALIDAR (2026-11-21) são pares (pedido anterior,
mensagem): a mensagem continua essa pesquisa e não pode receber «A». Em produção,
«e japonesas?» depois de «Encontra peças chinesas» recebeu «A» e passou a «obras
de Jean-Baptiste Pillement», o autor do primeiro cartão.

    python scripts/medir_autor_da_peca.py x3
"""

PECAS = [
    ("Caridade", "Joaquim Machado de Castro"),
    ("Tentações de Santo Antão", "Jheronymus Bosch"),
    ("São Jerónimo", "Albrecht Dürer"),
    ("Palanque chinês", "Jean-Baptiste Pillement"),
]
# o pedido que podia ter posto cada peça à frente do visitante
ANTES_DA_PECA = {
    "Caridade": "Mostra-me esculturas de barro",
    "Tentações de Santo Antão": "Há pinturas com demónios?",
    "São Jerónimo": "pinturas de santos",
    "Palanque chinês": "Encontra peças chinesas",
}

OBRAS_DO_AUTOR = [
    "Que mais obras existem deste autor?",
    "QUe mais obras existem deste autor?",
    "há mais peças do mesmo autor?",
    "mostra-me outras obras dele",
    "o que mais fez este artista?",
    "que outras coisas fez quem fez isto?",
    "outras obras do autor",
    "tens mais coisas deste autor aqui no museu?",
    "quero ver mais do mesmo autor",
    "e este autor tem mais peças na visita?",
    "ele fez mais alguma coisa que eu possa ver?",
    "mais obras deste senhor",
    "quem foi este autor?",
    "fala-me do autor desta peça",
    "what else did this artist make?",
    "more works by the same author",
    "show me other pieces by this artist",
    "do you have anything else by him?",
]

SOBRE_A_PECA = [
    "O que é isto?",
    "quem fez isto?",
    "quem é o autor?",
    "de que século é?",
    "de que material é feita?",
    "o que representa?",
    "fala-me mais desta peça",
    "é tesouro nacional?",
    "onde está na visita?",
    "qual é o número de inventário?",
    "porque é que é importante?",
    "de onde veio?",
    "que tamanho tem?",
    "who made this?",
    "what is it made of?",
    "tell me more about it",
]

OUTRA_COISA = [
    "pinturas com barcos",
    "Há obras de Gregório Lopes?",
    "mostra-me obras de Zurbarán",
    "Quem foi Josefa de Óbidos?",
    "há outras peças parecidas com esta?",
    "mostra-me esculturas do século XVIII",
    "Leva-me à Sala dos Presépios",
    "Que salas tem esta visita?",
    "faz-me um percurso de tesouros nacionais",
    "Olá",
    "obrigado",
    "qual é o horário do museu?",
    "Onde está a Mona Lisa?",
    "peças de vidro transparente",
    "tesouros nacionais de ourivesaria",
    "show me Japanese folding screens",
]

# (pedido anterior do visitante, mensagem): a mensagem continua a pesquisa anterior
SEGUIMENTOS_AFINAR = [
    ("Encontra peças chinesas", "e japonesas?"),
    ("Mostra-me peças de ouro", "e de prata?"),
    ("Há pinturas de santos?", "E esculturas?"),
    ("Peças do século XVII", "e do século XVIII?"),
    ("Mostra-me as peças mais recentes", "e mais antigas?"),
    ("Há desenhos de paisagens?", "e pinturas?"),
    ("Que tesouros nacionais há?", "e na visita?"),
    ("Encontra peças chinesas", "e indianas?"),
    ("Peças em madeira", "e em marfim?"),
    ("Mostra-me porcelanas", "mais"),
]
SEGUIMENTOS_VALIDAR = [
    ("Encontra peças japonesas", "e chinesas?"),
    ("Mostra-me peças de prata", "e de ouro?"),
    ("Há pinturas de paisagens?", "E desenhos?"),
    ("Peças do século XV", "e do século XVI?"),
    ("Mostra-me as peças mais antigas", "e mais recentes?"),
    ("Há faianças?", "e azulejos?"),
    ("O que há na visita?", "e nesta sala?"),
    ("Peças espanholas", "e portuguesas?"),
    ("Esculturas em pedra", "e em madeira?"),
    ("Mostra-me pinturas pequenas", "e maiores?"),
]
