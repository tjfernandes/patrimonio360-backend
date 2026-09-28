# -*- coding: utf-8 -*-
"""
Corrige aspas curvas nos scripts de accao de um export 3DVista.

Quando alguem escreve a accao do hotspot num editor com correccao automatica,
as aspas normais viram " e ", e o player rebenta com
    SyntaxError: illegal character U+201C
ao fazer eval do postMessage. O POI abre a janela mas nunca avisa o assistente.

    python fix_smart_quotes.py tours/mnaz            # mostra o que faria
    python fix_smart_quotes.py tours/mnaz --apply    # grava (guarda .bak)

Mexe apenas em script*.js e so troca aspas curvas por aspas escapadas.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CURVAS = {
    "“": '\\"',   # "
    "”": '\\"',   # "
}
SCRIPTS = ("script.js", "script_general.js", "script_mobile.js")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tour_dir", help="pasta do export 3DVista")
    ap.add_argument("--apply", action="store_true", help="grava as alteracoes")
    args = ap.parse_args()

    tour_dir = Path(args.tour_dir)
    total = 0

    for name in SCRIPTS:
        path = tour_dir / name
        if not path.exists():
            continue
        texto = path.read_text(encoding="utf-8")
        ocorrencias = sum(texto.count(c) for c in CURVAS)
        if not ocorrencias:
            print("{:20} limpo".format(name))
            continue

        total += ocorrencias
        print("{:20} {} aspas curvas".format(name, ocorrencias))

        if args.apply:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
            for curva, recta in CURVAS.items():
                texto = texto.replace(curva, recta)
            path.write_text(texto, encoding="utf-8")
            print("{:20} corrigido (original em {}.bak)".format("", name))

    if not total:
        print("\nNada a fazer.")
    elif args.apply:
        print("\n{} aspas corrigidas. Regenera o mapeamento a seguir:".format(total))
        print("    python tools/poi_mapping/extract_poi_mapping.py {} -o ...".format(tour_dir))
    else:
        print("\n{} aspas a corrigir. Volta a correr com --apply.".format(total))
        print("Nota: isto corrige o export. Se o tour for re-exportado do 3DVista")
        print("sem corrigir a accao no projecto, o problema volta.")


if __name__ == "__main__":
    main()
