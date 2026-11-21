"""Casos de medição da peça implícita pelo modelo (CHAT_IMPLICIT_ARTIFACT_MODEL).

Não é um teste automático: precisa do modelo real. De que peça fala a mensagem, entre as
que o ecrã mostra? Cada caso é (cenário, mensagem, artifact_id esperado ou None). None quer
dizer que a mensagem não é sobre nenhuma delas: pesquisa nova, outro assunto, ou uma peça
que não está no ecrã.

Um cenário tem os cartões visíveis, pela ordem do ecrã, e a peça em foco (`foco`): o
primeiro resultado, ou a peça aberta na visita (`source: tour_hotspot`). Um cartão com
`conjunto` e `membros` é o cartão de um conjunto: as outras peças do conjunto aparecem no
ecrã por baixo dele, mas o frontend só envia o cartão.

AFINAR serviu para escolher a redação do prompt; VALIDAR só foi corrido no fim; CONFIRMAR foi
escrito depois de tudo fechado. Mede-se com scripts/medir_peca_implicita.py. Os casos do
cenário «traje» são os de tests/test_implicit_artifact.py (TUNE e VALID), que as regras
acertam todos. SEGUIMENTOS_AFINAR e SEGUIMENTOS_VALIDAR (2026-11-21) são mensagens que mudam
de assunto, quase todas com um só cartão no ecrã. Resultados em
docs/peca-implicita-pelo-modelo.md. Qualquer alteração ao prompt repete a medição.
"""

TRAJE = [
    {"artifact_id": "a1", "inventory_number": "19663", "title": "Barrete de fumador/Masculino"},
    {"artifact_id": "a2", "inventory_number": "5622", "title": "Vestido (duas peças)/Feminino"},
    {"artifact_id": "a3", "inventory_number": "28519/1", "title": "Vestido/Criança"},
    {"artifact_id": "a4", "inventory_number": "26510", "title": "Retrato de família"},
    {"artifact_id": "a5", "inventory_number": "5262", "title": "Leque/Feminino"},
]
RETRATO_ABERTO = {"artifact_id": "h1", "inventory_number": "11459", "title": "Retrato de senhora", "source": "tour_hotspot"}
PAINEIS = [
    {"artifact_id": "v1", "inventory_number": "1550 Esc", "title": "São Vicente"},
    {"artifact_id": "v2", "inventory_number": "1082 Esc", "title": "São Vicente"},
    {"artifact_id": "p2", "inventory_number": "1362 Pint", "title": "Painel da Relíquia", "conjunto": "Painéis de São Vicente de Fora",
     "membros": [
         {"artifact_id": "p1", "inventory_number": "1361 Pint", "title": "Painel do Infante"},
         {"artifact_id": "p4", "inventory_number": "1364 Pint", "title": "Painel do Arcebispo"},
         {"artifact_id": "p3", "inventory_number": "1363 Pint", "title": "Painel dos Frades"},
         {"artifact_id": "p5", "inventory_number": "1365 Pint", "title": "Painel dos Cavaleiros"},
         {"artifact_id": "p6", "inventory_number": "1366 Pint", "title": "Painel dos Pescadores"},
     ]},
]
CASTRO = [
    {"artifact_id": "m1", "inventory_number": "60 Esc", "title": "Fé"},
    {"artifact_id": "m2", "inventory_number": "62 Esc", "title": "Gratidão"},
    {"artifact_id": "m3", "inventory_number": "90 Esc", "title": "Santa Teresa de Ávila"},
    {"artifact_id": "m4", "inventory_number": "57 Esc", "title": "Devoção"},
    {"artifact_id": "m5", "inventory_number": "63 Esc", "title": "Caridade"},
    {"artifact_id": "m6", "inventory_number": "59 Esc", "title": "Liberalidade"},
    {"artifact_id": "m7", "inventory_number": "584 Esc", "title": None},
]
MNSR = [
    {"artifact_id": "s1", "inventory_number": "166 Esc", "title": "O Desterrado"},
    {"artifact_id": "s2", "inventory_number": "110 Esc", "title": "Flor Agreste"},
    {"artifact_id": "s3", "inventory_number": "89 Esc", "title": "Conde de Ferreira"},
    {"artifact_id": "s4", "inventory_number": "120 Esc", "title": "Busto da Inglesa"},
]
CAPRI_ABERTO = {"artifact_id": "h2", "inventory_number": "107 Pin", "title": "Casas Brancas de Capri", "source": "tour_hotspot"}
AZULEJO = [
    {"artifact_id": "z1", "inventory_number": "1680", "title": "Painel de azulejos"},
    {"artifact_id": "z2", "inventory_number": "6341", "title": "Painel de azulejos"},
    {"artifact_id": "z3", "inventory_number": "143", "title": "Azulejo de padrão"},
    {"artifact_id": "z4", "inventory_number": "1", "title": "Grande Panorama de Lisboa"},
    {"artifact_id": "z5", "inventory_number": "902", "title": "Painel de azulejos"},
]


OURIVESARIA = [
    {"artifact_id": "o1", "inventory_number": "740 Our", "title": "Custódia de Belém"},
    {"artifact_id": "o2", "inventory_number": "540 Our", "title": "Cruz processional"},
    {"artifact_id": "o3", "inventory_number": "89 Our", "title": "Cálice"},
    {"artifact_id": "o4", "inventory_number": "1827 Our", "title": "Terrina", "conjunto": "Baixela Germain",
     "membros": [
         {"artifact_id": "o5", "inventory_number": "1830 Our", "title": "Saleiro"},
         {"artifact_id": "o6", "inventory_number": "1828 Our", "title": "Centro de mesa"},
         {"artifact_id": "o8", "inventory_number": "1832 Our", "title": "Cafeteira"},
     ]},
]
PINTURA = [
    {"artifact_id": "t1", "inventory_number": "1498 Pint", "title": "Tentações de Santo Antão"},
    {"artifact_id": "t2", "inventory_number": "828 Pint", "title": "São Jerónimo"},
    {"artifact_id": "t3", "inventory_number": "2150 Pint", "title": "Adoração dos Magos"},
    {"artifact_id": "t4", "inventory_number": "1165 Pint", "title": "Retrato de D. Sebastião"},
]
ANUNCIACAO_ABERTA = {"artifact_id": "h3", "inventory_number": "1211 Pint", "title": "Anunciação", "source": "tour_hotspot"}


BOSCH = [{"artifact_id": "b1", "inventory_number": "1498 Pint", "title": "Tentações de Santo Antão"}]
CUSTODIA = [{"artifact_id": "c1", "inventory_number": "740 Our", "title": "Custódia de Belém"}]
CHINESAS = [
    {"artifact_id": "k1", "inventory_number": "3106 Des", "title": "Palanque chinês"},
    {"artifact_id": "k2", "inventory_number": "3103 Des", "title": "Palanque chinês"},
    {"artifact_id": "k3", "inventory_number": "3104 Des", "title": "Palanque chinês"},
    {"artifact_id": "k4", "inventory_number": "3105 Des", "title": "Palanque chinês"},
    {"artifact_id": "k5", "inventory_number": "7581 Cer", "title": "Prato"},
    {"artifact_id": "k6", "inventory_number": "1653 Mov", "title": "Mesa de costura"},
    {"artifact_id": "k7", "inventory_number": "896 Div", "title": "Fole"},
    {"artifact_id": "k8", "inventory_number": "6710 Cer", "title": "Pote"},
]


def _primeiro(cartoes):
    return {key: cartoes[0][key] for key in ("artifact_id", "inventory_number", "title")} | {"source": "first_result"}


CENARIOS = {
    "traje": {"museu": "mnt", "visiveis": TRAJE, "foco": _primeiro(TRAJE)},
    "traje_aberta": {"museu": "mnt", "visiveis": TRAJE, "foco": RETRATO_ABERTO},
    "so_aberta": {"museu": "mnt", "visiveis": [], "foco": RETRATO_ABERTO},
    "paineis": {"museu": "mnaa", "visiveis": PAINEIS, "foco": _primeiro(PAINEIS)},
    "castro": {"museu": "mnaa", "visiveis": CASTRO, "foco": _primeiro(CASTRO)},
    "mnsr_aberta": {"museu": "mnsr", "visiveis": MNSR, "foco": CAPRI_ABERTO},
    "azulejo": {"museu": "mnaz", "visiveis": AZULEJO, "foco": _primeiro(AZULEJO)},
    "ourivesaria": {"museu": "mnaa", "visiveis": OURIVESARIA, "foco": _primeiro(OURIVESARIA)},
    "pintura_aberta": {"museu": "mnaa", "visiveis": PINTURA, "foco": ANUNCIACAO_ABERTA},
    "bosch_so": {"museu": "mnaa", "visiveis": BOSCH, "foco": _primeiro(BOSCH)},
    "custodia_so": {"museu": "mnaa", "visiveis": CUSTODIA, "foco": _primeiro(CUSTODIA)},
    "chinesas": {"museu": "mnaa", "visiveis": CHINESAS, "foco": _primeiro(CHINESAS)},
}

AFINAR = [
    # tests/test_implicit_artifact.py, TUNE
    ("traje", "Fala-me do segundo", "a2"), ("traje", "E o terceiro?", "a3"), ("traje", "O último é de quando?", "a5"),
    ("traje", "the second one", "a2"), ("traje", "E o 5622?", "a2"), ("traje", "Fala-me do 28519/1", "a3"),
    ("traje", "26510", "a4"), ("traje", "O retrato de família é de quando?", "a4"), ("traje", "Esse vestido é de seda?", "a2"),
    ("traje", "E o leque, de que é feito?", "a5"), ("traje", "De que é feito?", "a1"), ("traje", "Quem o usava?", "a1"),
    ("traje", "Mostra peças parecidas", "a1"), ("traje", "Conta-me a história", "a1"), ("traje", "Quero saber mais", "a1"),
    ("traje", "Encontra chapéus de mulher", None), ("traje", "Vestidos de noiva", None), ("traje", "Encontra o vestido de noiva", None),
    ("traje", "Retratos de família", None), ("traje", "Onde estou?", None), ("traje", "Que salas existem?", None), ("traje", "Olá", None),
    ("traje", "E o 12045?", None), ("traje", "Peças de 1850", None), ("traje", "Qual é o horário do museu?", None),
    ("traje", "O que é um tournure?", None), ("traje", "Quem foi o marquês de Angeja?", None),
    # peça aberta na visita (dos testes das regras)
    ("traje_aberta", "De que é feito?", "h1"), ("traje_aberta", "E o segundo?", "a2"), ("traje_aberta", "Encontra chapéus de mulher", None),
    ("traje_aberta", "Onde estou?", None), ("traje_aberta", "Procura leques", None), ("traje", "O que é esta peça?", "a1"),
    ("traje", "Fala-me desta obra", "a1"), ("so_aberta", "Que peça é?", "h1"), ("so_aberta", "What is this object?", "h1"),
    # conjunto: as peças do conjunto estão no ecrã, por baixo do cartão
    ("paineis", "O que representa o Painel dos Frades?", "p3"), ("paineis", "E o painel do infante?", "p1"),
    ("paineis", "o da relíquia é de quando?", "p2"), ("paineis", "E o segundo?", "v2"), ("paineis", "De que é feito?", "v1"),
    ("paineis", "Mostra-me pintura flamenga", None), ("paineis", "Quem foi Nuno Gonçalves?", None), ("paineis", "Onde estou?", None),
    ("paineis", "a escultura 1082 Esc é de que material?", "v2"), ("paineis", "Fala-me do Painel dos Pescadores", "p6"),
    ("paineis", "E o painel dos cavaleiros, está na visita?", "p5"),
    ("castro", "Fala-me da Caridade", "m5"), ("castro", "E a Liberalidade?", "m6"), ("castro", "A Santa Teresa é de barro?", "m3"),
    ("castro", "E a última?", "m7"), ("castro", "o 584 Esc o que é?", "m7"), ("castro", "É de barro?", "m1"),
    ("castro", "Quem foi Machado de Castro?", None), ("castro", "Há mais esculturas dele?", None),
    ("mnsr_aberta", "De que é feito?", "h2"), ("mnsr_aberta", "Quem pintou isto?", "h2"), ("mnsr_aberta", "E o Desterrado?", "s1"),
    ("mnsr_aberta", "o segundo é de mármore?", "s2"), ("mnsr_aberta", "Onde estou?", None), ("mnsr_aberta", "Mostra-me pinturas de Pousão", None),
    ("mnsr_aberta", "É de Capri?", "h2"),
    ("azulejo", "E o 6341?", "z2"), ("azulejo", "O Grande Panorama de Lisboa é de quando?", "z4"), ("azulejo", "o terceiro é holandês?", "z3"),
    ("azulejo", "Azulejos de padrão do século XVII", None), ("azulejo", "Quem o pintou?", "z1"), ("azulejo", "E o quinto?", "z5"),
]

VALIDAR = [
    # tests/test_implicit_artifact.py, VALID
    ("traje", "E a primeira?", "a1"), ("traje", "Fala do penúltimo", "a4"), ("traje", "O 3.º é de que época?", "a3"),
    ("traje", "E o 19663, quem o fez?", "a1"), ("traje", "Número 5262", "a5"), ("traje", "Do que é feito o barrete?", "a1"),
    ("traje", "Esse leque é pintado?", "a5"), ("traje", "E o vestido de criança?", "a3"), ("traje", "De que ano é?", "a1"),
    ("traje", "Tem assinatura?", "a1"), ("traje", "Há mais parecidos?", "a1"), ("traje", "É de seda?", "a1"),
    ("traje", "Procura leques", None), ("traje", "Mostra retratos", None), ("traje", "E sapatos?", None), ("traje", "Obrigado", None),
    ("traje", "Leva-me ao pátio", None), ("traje", "Quero ver pinturas do século XIX", None), ("traje", "Há vestidos de 1900?", None),
    ("traje", "Trajes de criança", None),
    ("traje_aberta", "Que salas existem?", None), ("traje_aberta", "Olá", None), ("traje_aberta", "Peças de 1850", None),
    ("traje", "Que peça é?", "a1"), ("traje", "What is this object?", "a1"), ("so_aberta", "O que é esta peça?", "h1"),
    ("so_aberta", "Fala-me desta obra", "h1"), ("so_aberta", "De que é feito?", "h1"),
    ("paineis", "Quem são os frades do painel dos frades?", "p3"), ("paineis", "O painel do arcebispo é de que século?", "p4"),
    ("paineis", "E o primeiro?", "v1"), ("paineis", "1361 Pint", "p1"), ("paineis", "Conta-me mais sobre o Painel da Relíquia", "p2"),
    ("paineis", "Há outros painéis no museu?", None), ("paineis", "Obrigado", None), ("paineis", "O dos pescadores tem quantas figuras?", "p6"),
    ("paineis", "De quando é?", "v1"), ("paineis", "Procura esculturas de santos", None),
    ("castro", "A Gratidão é de que ano?", "m2"), ("castro", "O que representa a Fé?", "m1"), ("castro", "E a terceira?", "m3"),
    ("castro", "57 Esc", "m4"), ("castro", "Quem a encomendou?", "m1"), ("castro", "Esculturas de barro do século XVIII", None),
    ("castro", "Que tamanho tem a Caridade?", "m5"),
    ("mnsr_aberta", "Em que ano foi pintado?", "h2"), ("mnsr_aberta", "A Flor Agreste é de quem?", "s2"), ("mnsr_aberta", "E o último?", "s4"),
    ("mnsr_aberta", "Que salas existem?", None), ("mnsr_aberta", "Procura retratos", None),
    ("mnsr_aberta", "Quem está retratado no Busto da Inglesa?", "s4"), ("mnsr_aberta", "Conta-me a história desta obra", "h2"),
    ("azulejo", "Fala-me do panorama de Lisboa", "z4"), ("azulejo", "inventário 902", "z5"), ("azulejo", "E o azulejo de padrão?", "z3"),
    ("azulejo", "Painéis de azulejos com cenas de caça", None), ("azulejo", "É português?", "z1"), ("azulejo", "the last one", "z5"),
    ("azulejo", "What is the fourth one?", "z4"),
]

# Escritos depois de fechada a redação e a ordem dos passos, para confirmar que não ficou afinada aos casos de cima.
CONFIRMAR = [
    ("ourivesaria", "A cafeteira é de prata?", "o8"), ("ourivesaria", "E o centro de mesa?", "o6"),
    ("ourivesaria", "Quem fez a Custódia de Belém?", "o1"), ("ourivesaria", "Há mais cálices?", None),
    ("ourivesaria", "o cálice tem pedras?", "o3"), ("ourivesaria", "E a terceira?", "o3"), ("ourivesaria", "1830 Our", "o5"),
    ("ourivesaria", "Cruzes processionais", None), ("ourivesaria", "Quanto pesa?", "o1"),
    ("ourivesaria", "Mostra-me a baixela Germain completa", None), ("ourivesaria", "Que rei encomendou a custódia?", "o1"),
    ("ourivesaria", "Pratas francesas do século XVIII", None), ("ourivesaria", "O saleiro era usado para quê?", "o5"),
    ("ourivesaria", "Bom dia", None), ("ourivesaria", "Quem foi Gil Vicente?", None),
    ("pintura_aberta", "Quem pintou as Tentações de Santo Antão?", "t1"), ("pintura_aberta", "O São Jerónimo é do Dürer?", "t2"),
    ("pintura_aberta", "De que ano é?", "h3"), ("pintura_aberta", "E o retrato de D. Sebastião?", "t4"),
    ("pintura_aberta", "Há outras Anunciações?", None), ("pintura_aberta", "A Anunciação é de quem?", "h3"),
    ("pintura_aberta", "Mostra-me pintura flamenga", None), ("pintura_aberta", "o segundo está na visita?", "t2"),
    ("pintura_aberta", "Que santo é este?", "h3"), ("pintura_aberta", "Pinturas de santos", None),
    ("pintura_aberta", "A adoração dos magos tem quantas figuras?", "t3"),
]

# Seguimentos que mudam de assunto (2026-11-21). Em produção, com um só cartão no ecrã («Tentações de Santo
# Antão»), «E gregório Lopes?» e «fala-me de gregório lopes» foram lidas como esse cartão: nos grupos de cima não
# havia nenhum ecrã com uma só peça nem mensagens com o nome de uma pessoa ou de uma peça que não está no ecrã.
SEGUIMENTOS_AFINAR = [
    ("bosch_so", "E gregório Lopes?", None), ("bosch_so", "fala-me de gregório lopes", None), ("bosch_so", "E do Dürer?", None),
    ("bosch_so", "As tentações são de que ano?", "b1"), ("bosch_so", "De que é feito?", "b1"), ("bosch_so", "e esculturas?", None),
    ("bosch_so", "E a Custódia de Belém?", None),
    ("custodia_so", "E a cruz de D. Sancho?", None), ("custodia_so", "A custódia é de ouro?", "c1"), ("custodia_so", "e de prata?", None),
    ("custodia_so", "E o Mestre João?", None),
    ("chinesas", "e japonesas?", None), ("chinesas", "E o fole?", "k7"), ("chinesas", "o prato é de porcelana?", "k5"),
    ("chinesas", "E biombos?", None), ("chinesas", "E gregório lopes?", None),
    ("castro", "E Soares dos Reis?", None), ("paineis", "E o Bosch?", None),
]
SEGUIMENTOS_VALIDAR = [
    ("bosch_so", "E Nuno Gonçalves?", None), ("bosch_so", "fala-me do Grão Vasco", None), ("bosch_so", "e pinturas flamengas?", None),
    ("bosch_so", "O tríptico das tentações está na visita?", "b1"), ("bosch_so", "Quem o pintou?", "b1"),
    ("bosch_so", "E os painéis de São Vicente?", None),
    ("custodia_so", "E o cálice de Alcobaça?", None), ("custodia_so", "Quanto pesa a custódia?", "c1"), ("custodia_so", "e relicários?", None),
    ("custodia_so", "fala-me de D. Manuel", None),
    ("chinesas", "e indianas?", None), ("chinesas", "A mesa de costura é de marfim?", "k6"), ("chinesas", "E o pote, de quando é?", "k8"),
    ("chinesas", "e do Japão?", None), ("chinesas", "fala-me de Pillement", None), ("chinesas", "E leques?", None),
    ("castro", "E a Josefa de Óbidos?", None), ("ourivesaria", "E o Germain?", None),
]

# Com uma peça aberta na visita, as regras dão-lhe prioridade mesmo quando a pergunta traz um assunto
# próprio. Medem-se à parte: são uma escolha, não uma leitura da mensagem.
ABERTA_COM_ASSUNTO = [
    ("traje_aberta", "Quem foi o marquês de Angeja?", "h1"), ("traje_aberta", "Qual é a técnica do azulejo?", "h1"),
    ("traje_aberta", "De que século é o painel?", "h1"),
]
