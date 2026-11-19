"""Casos do autor da peça em foco (CHAT_FOCUSED_ARTIFACT_AUTHOR).

O visitante tem uma peça à frente no minichat e a mensagem não diz o nome do
autor. OBRAS_DO_AUTOR devem receber «A» (o pedido passa a «obras de <autor>»);
SOBRE_A_PECA e OUTRA_COISA não devem receber «A» (seguem como hoje). Cada
mensagem é medida com as três peças de PECAS.

    python scripts/medir_autor_da_peca.py x3
"""

PECAS = [
    ("Caridade", "Joaquim Machado de Castro"),
    ("Tentações de Santo Antão", "Jheronymus Bosch"),
    ("São Jerónimo", "Albrecht Dürer"),
]

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
