# -*- coding: utf-8 -*-
"""
Corrige projectos 3DVista (.vtp) vindos de macOS cujos caminhos usam acentos
decompostos (NFD: «u» + acento combinante). No Windows «Conteúdos» em NFD e
«Conteúdos» em NFC sao pastas diferentes, e o 3DVista diz que os ficheiros
nao existem («File Missing ... Conteú dos ...»).

    python fix_vtp_unicode.py "Visitas/MNAA/3DVista - MNAA.vtp"            # so mostra
    python fix_vtp_unicode.py "Visitas/MNAA/3DVista - MNAA.vtp" --apply    # grava (faz .bak)

Normaliza todo o script.js para NFC. Nao muda mais nada.
"""
import argparse
import datetime as dt
import os
import shutil
import unicodedata
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vtp")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    vtp = Path(a.vtp)

    with zipfile.ZipFile(vtp) as z:
        text = z.read("script.js").decode("utf-8")
        entries = [(i, z.read(i.filename)) for i in z.infolist()]

    combining = sum(1 for ch in text if unicodedata.combining(ch))
    fixed = unicodedata.normalize("NFC", text)
    print(f"caracteres decompostos (NFD) no script.js: {combining}")
    if fixed == text:
        print("Nada a corrigir."); return
    if not a.apply:
        print("Volta a correr com --apply para gravar."); return

    tmp = vtp.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w") as out:
        for info, data in entries:
            out.writestr(info, fixed.encode("utf-8") if info.filename == "script.js" else data,
                         compress_type=info.compress_type)
    b = vtp.with_name(vtp.name + ".bak-" + dt.datetime.now().strftime("%Y%m%d-%H%M"))
    shutil.copy2(vtp, b)
    try:
        os.replace(tmp, vtp)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        b.unlink(missing_ok=True)
        raise SystemExit(f"Nao consegui gravar {vtp.name}: o projecto esta aberto no 3DVista. "
                         "Fecha-o (na caixa 'File Missing' carrega em No) e volta a correr.")
    print(f"gravado. copia do original: {b.name}")


if __name__ == "__main__":
    main()
