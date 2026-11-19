"""Casos da resposta a perguntas pelo aspeto (CHAT_VISUAL_ANSWER_RULE).

Cada caso é um pedido real do MNAA com os cartões que a pesquisa mostrou a
2026-11-19 (n.º de inventário). «grupos» são as coisas que o pedido nomeia,
cada uma com as palavras que a dizem (expressão regular, sem acentos nem
maiúsculas). A medição conta as vezes em que a resposta diz de uma peça uma
dessas coisas sem o cartão dessa peça a dizer.

    python scripts/medir_resposta_visual.py x5
"""

CASOS = [
    {"pedido": "retratos de senhoras com colares de pérolas",
     "inventarios": ["148 Min", "106 Min", "2201 Pint", "1981 Pint", "1731 Pint"],
     "grupos": {"colar": r"colar|gargantilha", "pérola": r"perola"}},
    {"pedido": "Encontra pinturas de mulheres com colares",
     "inventarios": ["100 Min", "1492 Pint", "1731 Pint", "91 Min", "2201 Pint", "1981 Pint"],
     "grupos": {"colar": r"colar|gargantilha"}},
    {"pedido": "retratos de senhoras com colares",
     "inventarios": ["100 Min", "91 Min", "159 Min", "1460 Pint", "148 Min", "106 Min", "2201 Pint", "1981 Pint"],
     "grupos": {"colar": r"colar|gargantilha"}},
    {"pedido": "encontra estatuas brancas",
     "inventarios": ["1195 Esc", "539 Esc", "1019 Esc"],
     "grupos": {"branco": r"branc|alv[oa]s?\b"}},
    {"pedido": "esculturas em mármore branco",
     "inventarios": ["745 Esc", "761 Esc", "53 Esc", "762 Esc"],
     "grupos": {"mármore": r"marmore", "branco": r"branc"}},
    {"pedido": "pinturas com barcos",
     "inventarios": ["468 Pint", "1231 Pint", "749 Pint", "467 Pint", "3267 Des", "14220 Grav", "1560 Mov"],
     "grupos": {"barco": r"barc|navio|\bnaus?\b|galea|caravela|embarcac|veleiro"}},
    {"pedido": "tecidos vermelhos com fios de ouro",
     "inventarios": ["1918 Tec", "614 Tec", "4572 Tec"],
     "grupos": {"vermelho": r"vermelh|carmes|encarnad|escarlate", "ouro": r"ouro|dourad"}},
    {"pedido": "quadros escuros com uma vela acesa",
     "inventarios": ["538 Pint", "1629 Pint", "13/16v. Ilum", "3304 Des", "118 Pint"],
     "grupos": {"vela": r"\bvelas?\b|cirio|candel|castical|lampada|lamparina|candeia"}},
    {"pedido": "quadros com caveiras",
     "inventarios": ["1629 Pint", "772 Pint", "14226 Grav", "14/234v. Ilum", "1174 Pint", "98 Pint", "1100 Esc", "1066 Pint"],
     "grupos": {"caveira": r"caveira|cranio|esqueleto|cabeca de morto"}},
    {"pedido": "móveis com embutidos de madrepérola",
     "inventarios": ["1264 Mov", "1265 Mov", "1550 Mov", "18 Cx", "1666 Mov", "824 Mov", "823 Mov", "99 Cx"],
     "grupos": {"madrepérola": r"madreperola|nacar"}},
    {"pedido": "peças de vidro transparente",
     "inventarios": ["971 Vid", "194 Vid", "931 Vid", "964 Vid", "1094 Vid", "37 Vid", "1352 Vid", "232 Vid"],
     "grupos": {"transparente": r"transparen|incolor|translucid"}},
    {"pedido": "santos com um livro na mão",
     "inventarios": ["14205 Grav", "646 Esc", "1336 Esc", "14216 Grav", "1522 Esc", "990 Esc", "1080 Esc", "1143 Esc"],
     "grupos": {"livro": r"livro|evangelh|biblia|breviario|missal"}},
    {"pedido": "pinturas com anjos",
     "inventarios": ["1020 Pint", "30 Pint", "29 Pint", "532 Pint", "9 Pint", "1466 Pint", "677 Pint", "1277 Pint"],
     "grupos": {"anjo": r"anjo|querubi|serafi|putt[oi]|arcanjo|angelical"}},
    {"pedido": "blue and white porcelain", "lingua": "en",
     "inventarios": ["7583 Cer", "2103 Cer", "1924 Cer", "268 Cer", "6591 Cer", "4289 Cer", "6710 Cer", "6738 Cer"],
     "grupos": {"azul": r"azul|blue|cobalt", "branco": r"branc|white"}},
]
