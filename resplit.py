"""Refaz o split treino/validacao do yolo_dataset por BLOCO TEMPORAL.

  ./venv/bin/python resplit.py --dry-run     # mostra o plano, nao move nada
  ./venv/bin/python resplit.py               # aplica (grava manifesto do split atual)
  ./venv/bin/python resplit.py --reverter    # volta ao split anterior

POR QUE NAO SORTEAR IMAGENS
As imagens vem de uma camera bullet fixa na ponta do guindaste. Os nomes carregam
timestamp (bullet_AAAAMMDD_HHMMSS) e, dentro de uma sessao de operacao, existem
quadros separados por poucos segundos -- medido neste dataset: lacuna minima de
1 s, mediana de 20 s, e 85 dos 158 pares consecutivos abaixo de 30 s. Sortear
imagem por imagem coloca quadros quase identicos nos dois lados, o modelo decora
e a metrica mente para cima.

O QUE ESTE SCRIPT FAZ
1. Agrupa as imagens em sessoes: mesmo dia e lacuna <= --limiar segundos.
2. Para cada dia, escolhe por busca exaustiva o subconjunto de sessoes que
   deixa a validacao mais perto de --frac, garantindo que sobre pelo menos uma
   sessao no treino. Assim TODOS os dias aparecem no treino -- o split antigo
   separava dias inteiros e o treino nunca via 2 das 4 condicoes.
3. Move imagens e labels, e apaga os .cache do Ultralytics (obrigatorio: eles
   guardam a lista de arquivos do split velho) e os .npy stale do antigo
   cache='disk', que nao sao mais lidos porque o train.py usa cache=False.
4. Confere e reporta a menor distancia temporal entre uma imagem de treino e uma
   de validacao no mesmo dia. Esse numero e a garantia de que nao houve
   vazamento; ele tem que ser >= --limiar.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

RAIZ = Path("yolo_dataset")
MANIFESTO = RAIZ / "_split_anterior.json"
PADRAO = re.compile(r"bullet_(\d{8})_(\d{6})")
NOMES = {0: "load", 1: "person", 2: "pipe"}


def coletar() -> list[dict]:
    itens = []
    for split in ("train", "val"):
        for img in sorted((RAIZ / "images" / split).glob("*.jpg")):
            m = PADRAO.search(img.name)
            if not m:
                raise SystemExit(f"sem timestamp no nome: {img.name}")
            lbl = RAIZ / "labels" / split / f"{img.stem}.txt"
            linhas = [l for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            itens.append({
                "ts": datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"),
                "split": split, "img": img, "lbl": lbl,
                "classes": [int(l.split()[0]) for l in linhas],
            })
    itens.sort(key=lambda d: d["ts"])
    return itens


def sessoes(itens: list[dict], limiar: int) -> dict:
    ses, k, ant = defaultdict(list), 0, None
    for it in itens:
        if ant is None or it["ts"].date() != ant.date() or (it["ts"] - ant).total_seconds() > limiar:
            k += 1
        ses[(it["ts"].date(), k)].append(it)
        ant = it["ts"]
    return ses


def escolher_val(ses: dict, frac: float) -> set:
    """Por dia, busca exaustiva do subconjunto de sessoes mais proximo de `frac`."""
    por_dia = defaultdict(list)
    for chave, itens in ses.items():
        por_dia[chave[0]].append(chave)
    escolhidas = set()
    for dia, chaves in sorted(por_dia.items()):
        total = sum(len(ses[c]) for c in chaves)
        pessoas = sum(c2 == 1 for c in chaves for it in ses[c] for c2 in it["classes"])
        melhor, melhor_score = (), float("inf")
        for n in range(1, len(chaves)):  # nunca todas: sobra >=1 sessao no treino
            for comb in combinations(sorted(chaves), n):
                n_img = sum(len(ses[c]) for c in comb)
                n_pes = sum(c2 == 1 for c in comb for it in ses[c] for c2 in it["classes"])
                score = abs(n_img / total - frac)
                if pessoas:
                    score += 0.5 * abs(n_pes / pessoas - frac)
                if score < melhor_score:
                    melhor, melhor_score = comb, score
        escolhidas |= set(melhor)
    return escolhidas


def resumo(itens: list[dict], destino: dict) -> None:
    for split in ("train", "val"):
        sel = [it for it in itens if destino[it["img"].name] == split]
        cls = defaultdict(int)
        for it in sel:
            for c in it["classes"]:
                cls[c] += 1
        vazias = sum(1 for it in sel if not it["classes"])
        dias = defaultdict(int)
        for it in sel:
            dias[str(it["ts"].date())] += 1
        print(f"  {split:5s} {len(sel):3d} imagens ({vazias} background, {vazias/len(sel):.0%})"
              f"  instancias: " + " ".join(f"{NOMES[c]}={cls[c]}" for c in sorted(cls)))
        print(f"        por dia: {dict(sorted(dias.items()))}")


def checar_vazamento(itens: list[dict], destino: dict, limiar: int) -> None:
    por_dia = defaultdict(lambda: ([], []))
    for it in itens:
        por_dia[it["ts"].date()][destino[it["img"].name] == "val"].append(it["ts"])
    pior, pior_dia = None, None
    for dia, (tr, va) in por_dia.items():
        if not tr or not va:
            print(f"  ATENCAO: dia {dia} nao aparece nos dois splits (train={len(tr)} val={len(va)})")
            continue
        d = min(abs((a - b).total_seconds()) for a in tr for b in va)
        if pior is None or d < pior:
            pior, pior_dia = d, dia
    if pior is not None:
        ok = "OK" if pior >= limiar else "FALHOU"
        print(f"  menor distancia treino<->validacao no mesmo dia: {pior:.0f}s "
              f"(limiar {limiar}s) -> {ok}")


def aplicar(itens: list[dict], destino: dict) -> None:
    MANIFESTO.write_text(json.dumps({it["img"].name: it["split"] for it in itens}, indent=1))
    movidos = 0
    for it in itens:
        novo = destino[it["img"].name]
        if novo == it["split"]:
            continue
        for sub, arq in (("images", it["img"]), ("labels", it["lbl"])):
            alvo = RAIZ / sub / novo / arq.name
            if arq.exists():
                alvo.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(arq), str(alvo))
        movidos += 1
    lixo = list(RAIZ.rglob("*.cache")) + list(RAIZ.rglob("*.npy"))
    for f in lixo:
        f.unlink()
    print(f"\n  {movidos} imagens movidas | {len(lixo)} arquivos de cache apagados")
    print(f"  manifesto do split anterior: {MANIFESTO}")


def reverter() -> None:
    if not MANIFESTO.exists():
        raise SystemExit("sem manifesto para reverter")
    antigo = json.loads(MANIFESTO.read_text())
    itens = coletar()
    n = 0
    for it in itens:
        alvo_split = antigo.get(it["img"].name)
        if alvo_split and alvo_split != it["split"]:
            for sub, arq in (("images", it["img"]), ("labels", it["lbl"])):
                if arq.exists():
                    shutil.move(str(arq), str(RAIZ / sub / alvo_split / arq.name))
            n += 1
    for f in list(RAIZ.rglob("*.cache")):
        f.unlink()
    print(f"  revertido: {n} imagens voltaram ao split anterior")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limiar", type=int, default=300,
                    help="segundos de lacuna que separam duas sessoes (default 300)")
    ap.add_argument("--frac", type=float, default=0.25, help="fracao alvo de validacao")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reverter", action="store_true")
    args = ap.parse_args()

    if args.reverter:
        return reverter()

    itens = coletar()
    ses = sessoes(itens, args.limiar)
    val_ses = escolher_val(ses, args.frac)
    destino = {it["img"].name: ("val" if chave in val_ses else "train")
               for chave, lista in ses.items() for it in lista}

    print(f"\n  {len(itens)} imagens, {len(ses)} sessoes (limiar {args.limiar}s), "
          f"{len(val_ses)} sessoes para validacao\n")
    print("  ANTES:")
    resumo(itens, {it["img"].name: it["split"] for it in itens})
    print("\n  DEPOIS:")
    resumo(itens, destino)
    print()
    checar_vazamento(itens, destino, args.limiar)

    if args.dry_run:
        print("\n  (dry-run: nada foi movido)\n")
    else:
        aplicar(itens, destino)


if __name__ == "__main__":
    main()
