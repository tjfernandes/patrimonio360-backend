"""Listas fechadas do enriquecimento por regras e os padrões que levam o texto da ficha até elas.

Os padrões aplicam-se a texto normalizado (minúsculas, sem acentos). Cada entrada é
(valor da lista, padrão); um valor pode ter um nível acima, separado por « › »
(«madeira › carvalho» também conta como «madeira»).

Versão mnaa-1: construída com os valores que existem nas fichas do MNAA e conferida
no piloto de 2026-11-24 (docs/enriquecimento-por-campos.md). Outro museu pode
precisar de valores novos (traje, azulejo, arqueologia): o construir.py lista o que
ficou sem correspondência.
"""
import re
import unicodedata

VERSAO = "mnaa-1"


def n(texto) -> str:
    if isinstance(texto, (list, tuple)):
        texto = " ; ".join(str(x) for x in texto if x)
    texto = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", texto).strip()


def _c(lista):
    return [(valor, re.compile(r"(?<![a-z])(?:" + padrao + r")(?![a-z])")) for valor, padrao in lista]


MATERIAIS = _c([
    ("madeira › carvalho", r"carvalho"), ("madeira › castanho", r"madeira de castanho|castanho entalhad[oa]"),
    ("madeira › pinho", r"pinho"), ("madeira › choupo", r"choupo"), ("madeira › nogueira", r"nogueira"),
    ("madeira › pau-santo", r"pau[- ]santo|jacaranda"), ("madeira › mogno", r"mogno"), ("madeira › ébano", r"ebano"),
    ("madeira › teca", r"teca"), ("madeira › sissó", r"sisso"), ("madeira › casquinha", r"casquinha"),
    ("madeira › cedro", r"cedro"), ("madeira › vinhático", r"vinhatico"), ("madeira", r"madeiras?"),
    ("prata dourada", r"prata dourada"), ("prata", r"prata"), ("ouro", r"ouro"), ("bronze", r"bronze"),
    ("cobre", r"cobre"), ("latão", r"latao"), ("ferro", r"ferro"), ("estanho", r"estanho"), ("aço", r"aco"),
    ("vidro", r"vidros?|cristal"), ("marfim", r"marfim"), ("osso", r"osso"), ("tartaruga", r"tartaruga"),
    ("madrepérola", r"madreperola"), ("coral", r"coral"), ("âmbar", r"ambar"), ("pérolas", r"perolas?|aljofar(?:es)?"),
    ("gemas", r"gemas?|diamantes?|rubis?|rubi|esmeraldas?|safiras?|granadas?|ametistas?|topazios?|pedras preciosas|pedraria"),
    ("esmalte", r"esmaltes?"), ("pedra › mármore", r"marmore"), ("pedra › alabastro", r"alabastro"),
    ("pedra › calcário", r"calcario|pedra de anca"), ("pedra › jaspe", r"jaspe"), ("pedra", r"pedra(?! preciosa)|granito"),
    ("cerâmica › terracota", r"terracota|barro(?: cozido)?"), ("cerâmica › faiança", r"faianca"),
    ("cerâmica › porcelana", r"porcelana"), ("cerâmica › grés", r"gres"),
    ("papel", r"papel|papeis"), ("pergaminho", r"pergaminho"), ("velino", r"velino"), ("cartão", r"cartao"),
    ("tela", r"tela"), ("seda", r"sedas?"), ("linho", r"linho"), ("algodão", r"algodao"), ("lã", r"la"),
    ("veludo", r"veludo"), ("cetim", r"cetim"), ("damasco", r"damasco"), ("brocado", r"brocad[oa]|brocatel"),
    ("tafetá", r"tafeta"), ("fio metálico", r"fios? (?:de )?(?:ouro|prata|metalicos?|metal)|fio laminado"),
    ("couro", r"couro|cabedal|pele"), ("laca", r"laca|charao"),
])

TECNICAS = _c([
    ("pintura a óleo", r"oleo"), ("pintura a têmpera", r"tempera"), ("iluminura", r"iluminura|iluminad[oa]s?"),
    ("aplicação de ouro", r"dourad[oa]s?|douramento|folha de ouro|a ouro|e ouro"), ("policromia", r"policromad[oa]s?|policromia"),
    ("estofado", r"estofad[oa]s?"), ("desenho", r"desenho"), ("desenho › pena", r"a pena|pena e"),
    ("desenho › aguada", r"aguadas?"), ("desenho › aguarela", r"aguarel\w+"), ("desenho › guache", r"gouache|guache"),
    ("desenho › carvão", r"carvao"), ("desenho › sanguínea", r"sanguinea"), ("desenho › lápis", r"lapis"),
    ("desenho › giz", r"giz"), ("desenho › pastel", r"pastel"),
    ("gravura › buril", r"buril"), ("gravura › água-forte", r"agua[- ]? ?forte"), ("gravura › ponteado", r"ponteado"),
    ("gravura › litografia", r"litografi\w+"), ("gravura › xilogravura", r"xilogravura"), ("gravura › água-tinta", r"agua[- ]tinta"),
    ("escultura de vulto", r"vulto"), ("relevo", r"relevos?|relevad[oa]s?"), ("talha", r"entalhad[oa]s?|talha|entalhe"),
    ("modelação", r"modelad[oa]s?"), ("fundição", r"fundid[oa]s?|fundicao"), ("cinzelado", r"cinzelad[oa]s?"),
    ("repuxado", r"repuxad[oa]s?"), ("incisão", r"incis[oa]s?"), ("filigrana", r"filigrana"), ("esmaltagem", r"esmaltad[oa]s?|esmaltes?"),
    ("torneado", r"tornead[oa]s?"), ("embutidos", r"embutid[oa]s?|marchetad[oa]s?|marchetaria"), ("lacagem", r"lacad[oa]s?|laca|charao"),
    ("vidro soprado", r"soprad[oa]s?"), ("lapidação", r"lapidad[oa]s?"), ("gravação", r"gravad[oa]s?"),
    ("moldagem", r"moldad[oa]s?"), ("decoração pintada", r"pintad[oa]s?"), ("vidrado", r"vidrad[oa]s?"), ("cozedura", r"cozid[oa]s?"),
    ("bordado", r"bordad[oa]s?"), ("tapeçaria", r"tapecaria"), ("renda", r"rendas?"),
    ("tecido lavrado", r"tecido lavrado|lavrad[oa]s? a|lampasso|damasco|brocatel"), ("estampagem", r"estampad[oa]s?"),
    ("couro lavrado", r"couro lavrado"), ("manuscrito", r"escrito a|manuscrit[oa]s?"),
])

ESTILOS = _c([
    ("gótico", r"gotic[oa]s?"), ("manuelino", r"manuelin[oa]s?"), ("renascimento", r"renascentistas?|renascimento"),
    ("maneirismo", r"maneiristas?|maneirismo"), ("barroco", r"barroc[oa]s?"), ("rococó", r"rococo|rocaille"),
    ("neoclássico", r"neoclassic[oa]s?|neoclassicismo"), ("romantismo", r"romantic[oa]s?|romantismo"),
    ("namban", r"namban"), ("indo-português", r"indo[- ]portugues\w*"), ("mudéjar", r"mudejar"),
    ("hispano-mourisco", r"hispano[- ]mouris\w+"), ("império", r"estilo imperio"),
    ("D. João V", r"estilo d\.? ?joao v"), ("D. José", r"estilo d\.? ?jose"), ("D. Maria", r"estilo d\.? ?maria(?!n)"),
    ("Luís XV", r"luis xv(?!i)"), ("Luís XVI", r"luis xvi"),
])

# lugar → (país ou grande região, região/cidade canónica)
LUGARES = [
    (r"lisboa", "Portugal", "Lisboa"), (r"porto", "Portugal", "Porto"), (r"arraiolos", "Portugal", "Arraiolos"),
    (r"castelo branco", "Portugal", "Castelo Branco"), (r"marinha grande", "Portugal", "Marinha Grande"), (r"covo", "Portugal", "Covo"),
    (r"elvas", "Portugal", "Elvas"), (r"coimbra", "Portugal", "Coimbra"), (r"evora", "Portugal", "Évora"), (r"viseu", "Portugal", "Viseu"),
    (r"braga", "Portugal", "Braga"), (r"aveiro", "Portugal", "Aveiro"), (r"portugal|portugues\w*", "Portugal", None),
    (r"la granja|granja de s", "Espanha", "La Granja de San Ildefonso"), (r"catalunha", "Espanha", "Catalunha"),
    (r"valencia", "Espanha", "Valência"), (r"almeria|andaluzia", "Espanha", "Andaluzia"), (r"sevilha", "Espanha", "Sevilha"),
    (r"toledo", "Espanha", "Toledo"), (r"talavera", "Espanha", "Talavera"), (r"madrid", "Espanha", "Madrid"),
    (r"espanha|espanhol\w*", "Espanha", None), (r"peni?n?sula iberica", "Península Ibérica", None),
    (r"paris", "França", "Paris"), (r"beauvais", "França", "Beauvais"), (r"aubusson", "França", "Aubusson"), (r"sevres", "França", "Sèvres"),
    (r"vincennes", "França", "Vincennes"), (r"limoges", "França", "Limoges"), (r"lyon", "França", "Lyon"), (r"franca|frances\w*", "França", None),
    (r"roma", "Itália", "Roma"), (r"veneza|murano", "Itália", "Veneza"), (r"florenca|toscan\w+", "Itália", "Toscana"),
    (r"milao", "Itália", "Milão"), (r"napoles", "Itália", "Nápoles"), (r"genova", "Itália", "Génova"), (r"italia|italian\w*", "Itália", None),
    (r"augsburg\w*|augustae vindelicorum", "Alemanha", "Augsburgo"), (r"meissen", "Alemanha", "Meissen"), (r"nuremberga", "Alemanha", "Nuremberga"),
    (r"alemanha|alema\w*", "Alemanha", None),
    (r"bruxelas", "Flandres e Países Baixos", "Bruxelas"), (r"antuerpia", "Flandres e Países Baixos", "Antuérpia"),
    (r"bruges", "Flandres e Países Baixos", "Bruges"), (r"malines", "Flandres e Países Baixos", "Malines"), (r"delft", "Flandres e Países Baixos", "Delft"),
    (r"flandres|flameng\w+|provincias unidas|holanda|holandes\w*|paises baixos", "Flandres e Países Baixos", None),
    (r"londres", "Inglaterra", "Londres"), (r"inglaterra|ingles\w*", "Inglaterra", None),
    (r"goa", "Índia", "Goa"), (r"cochim", "Índia", "Cochim"), (r"bengala", "Índia", "Bengala"), (r"guzarate", "Índia", "Guzarate"),
    (r"india(?!s)|indian\w+|indo[- ]portugues\w*", "Índia", None),
    (r"jingdezhen", "China", "Jingdezhen"), (r"macau", "China", "Macau"), (r"cantao", "China", "Cantão"), (r"china|chines\w*", "China", None),
    (r"japao|japones\w*|namban", "Japão", None), (r"persia\w*|persa", "Pérsia", None), (r"egipto|egipci\w+", "Egipto", None),
    (r"turquia|turc[oa]s?|otoman\w+", "Turquia", None), (r"ceilao|cingal\w+", "Ceilão", None), (r"serra leoa|sapi", "Serra Leoa", None),
    (r"benim", "Benim", None), (r"brasil\w*", "Brasil", None), (r"siria", "Síria", None), (r"grecia|greg[oa]s?", "Grécia", None),
]
LUGARES = [(re.compile(r"(?<![a-z])(?:" + p + r")(?![a-z])"), pais, regiao) for p, pais, regiao in LUGARES]
SEM_LUGAR = re.compile(r"desconhecid|nao determinad|indeterminad")

# função de uso, pelo tipo de objeto (palavra principal do título) e pela categoria
FUNCAO_POR_TIPO = {
    "litúrgico": "calice custodia casula dalmatica pluvial frontal pixide estola manipulo mitra relicario sacra turibulo naveta missal "
                 "galheta sacrario altar resplendor ambula caldeirinha ostensorio lampadario sebasto capuz".split(),
    "serviço de mesa": "copo prato garrafa travessa terrina saleiro colher garfo faca jarro caneca taca acucareiro galheteiro cafeteira bule "
                       "chavena molheira fruteiro compoteira doceira covilhete pires leiteira chaleira especieiro olha refrescador boiao "
                       "frasco porro porron cantir concha gomil bacia escalfador licoreira".split(),
    "iluminação": "castical lamparina candelabro tocheiro lanterna palmatoria".split(),
    "assento": "cadeira tamborete banco canape sofa cadeiral duchesse bergere faldistorio preguiceira arquibanco".split(),
    "arrumação": "arca cofre contador comoda armario bau caixa estojo arqueta gaveta arcaz estante aparador".split(),
    "escrita e estudo": "escritorio secretaria tinteiro escrivaninha papeleira".split(),
    "adorno pessoal": "pendente colar anel brincos adorno leque pente insignia pulseira alfinete broche fivela medalhao".split(),
    "vestuário": "gola lenco traje colete luvas sapatos chapeu vestido".split(),
    "decoração de interiores": "tapecaria colcha tapete alcatifa guarnicao jarra biombo espelho pano tremo consola vaso pote placa moldura".split(),
    "mobiliário de apoio": "mesa bufete toucador lavatorio leito cama tabuleiro maquineta oratorio".split(),
}
RELIGIOSO_TITULO = re.compile(
    r"(?<![a-z])(?:sao|s\.|santo|santa|santos|santas|santissima|virgem|cristo|jesus|nossa senhora|senhor|anjo|arcanjo|apostolos?|"
    r"adoracao|anunciacao|natividade|calvario|crucifi\w+|deposicao|descida da cruz|lamentacao|ressurreicao|ressureicao|ascensao|"
    r"assuncao|pentecostes|circuncisao|visitacao|epifania|baptismo|martirio|pieta|ecce homo|sagrada familia|trindade|"
    r"evangel\w+|salmos?|horas|oracao|oficio|ladainhas?|litanias?|litania|missa|antifona|cantico|profeta|madalena|presepio|eucaristia|imaculada)(?![a-z])")
MANUSCRITO_TITULO = re.compile(r"^(?:horas|folio|folios|oracao|calendario|salmos?|oficio|ladainhas?|litanias?|evangelho|missa|cantico|antifona|compromisso|livro)(?![a-z])")

CLASSES_TAMANHO = [(10, "miniatura (até 10 cm)"), (30, "pequeno (10 a 30 cm)"), (100, "médio (30 cm a 1 m)"), (200, "grande (1 a 2 m)"), (1e9, "monumental (mais de 2 m)")]

# Tipo de objeto: a palavra principal do título, nas categorias de objetos (vidros, mobiliário, ourivesaria,
# têxteis, cerâmica, diversos). Lista tirada dos títulos do MNAA (palavras com pelo menos duas peças), limpa à mão.
TIPOS = """adorno alcatifa altar anel aparador arca arcaz armário arqueta arquibanco açucareiro bacia banco baú bergère biombo
boião brincos bufete bule cadeira cadeiral cafeteira caixa canapé candelabro caneca cantir capuz castiçal casula cesto chaleira
chávena cofre colar colcha colher compoteira concha consola contador copo covilhete cruz custódia cálice cómoda dalmática doceira
duchesse entremeio escalfador escritório escrivaninha especieiro espelho estante estatueta estojo estola faca faldistório fragmento
frasco frontal fruteiro funil galheta galheteiro garfo garrafa gaveta gola gomil guarnição insígnia jarra jarro kendi lamparina
lavatório leiteira leito lenço leque licoreira manípulo maquineta medalhão mesa mitra molheira naveta olha olifante oratório painel
pano pendente pente pires placa pluvial porró pote prato preguiceira píxide refrescador relicário relógio resplendor sacra sacrário
saleiro salva sebasto secretária sofá tabuleiro tamborete tapete tapeçaria taça terrina tinteiro toucador travessa tremó vaso""".split()
TIPO_POR_CHAVE = {n(t): t for t in TIPOS}
TIPO_POR_CHAVE.update({"fragmentos": "fragmento", "guarnicoes": "guarnição", "porron": "porró", "brinco": "brincos"})
RAMO_DA_CATEGORIA = {"vidros": "vidro", "mobiliario": "mobiliário", "ourivesaria": "ourivesaria", "texteis": "têxtil", "ceramica": "cerâmica",
                     "diversos": "objeto", "metais": "metal", "traje e aderecos": "traje", "": "objeto"}

# Cores escritas na ficha. Parte da tabela do indexador das legendas, com três correções vistas no MNAA: «bordo»
# (a borda de um prato) não é vinoso, «laranja da China» é a fruta, e marfim e âmbar numa ficha são matérias.
CORES = _c([
    ("branco", r"branc[oa]s?"), ("preto", r"pret[oa]s?|negr[oa]s?"), ("azul", r"azu(?:l|is)"), ("verde", r"verdes?"),
    ("amarelo", r"amarel[oa]s?"), ("vermelho", r"vermelh[oa]s?|encarnad[oa]s?"), ("castanho", r"castanh[oa]s?"),
    ("cinzento", r"cinzent[oa]s?|cinza"), ("rosa", r"cor[- ]de[- ]rosa|rosad[oa]s?|(?<=seda )rosa|(?<=fundo )rosa|(?<=tons de )rosa"), ("roxo", r"rox[oa]s?|violeta|lilas|purpura"),
    ("laranja", r"laranja(?! da china)|alaranjad[oa]s?"), ("dourado", r"dourad[oa]s?|ouro"), ("prateado", r"pratead[oa]s?|prata"),
    ("bege", r"beges?|beije"), ("creme", r"creme"), ("ocre", r"ocres?"), ("manganês", r"manganes"), ("azul de cobalto", r"cobalto"),
    ("vinoso", r"vinos[oa]s?|bordeaux|cor de vinho"), ("turquesa", r"turquesa"), ("sépia", r"sepia"), ("carmim", r"carmim"),
    ("escarlate", r"escarlate"), ("polícromo", r"policrom[oa]s?"),
])
