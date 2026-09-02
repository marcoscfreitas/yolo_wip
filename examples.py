"""Gera imagens de exemplo com a inferencia desenhada, para inspecao visual.

  ./venv/bin/python examples.py              # os dois modelos, 8 imagens cada
  ./venv/bin/python examples.py --n 16
  ./venv/bin/python examples.py --conf 0.15  # afrouxa o limiar para ver o que ele quase pegou

Cada imagem de saida mostra apenas o que a rede detectou: a mascara preenchida na
cor da classe, o contorno e a confianca. Nada de anotacao sobreposta.
A contagem detectado/anotado fica no resumo.txt de cada pasta.

DUPLO PASSE POR RESOLUCAO
O sweep mostrou que as classes preferem resolucoes opostas de inferencia, e isso
se repetiu nos dois modelos treinados de forma independente:
    load   -> melhor em 1280 (subir a resolucao piora: sai da escala de treino)
    person -> melhor na mais alta (traz as pessoas de ~20 px para a faixa detectavel)
Entao cada modelo roda dois passes e cada classe vem do passe onde ela e melhor.
Como o mAP e calculado por classe de forma independente, isso e exatamente o
ganho que a tabela do sweep media.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# BGR
CORES = {0: (255, 140, 0), 1: (60, 220, 60), 2: (0, 215, 255)}
NOMES = {0: "load", 1: "person", 2: "pipe"}
BRANCO = (255, 255, 255)

MODELOS = {
    "res_1408": dict(pesos="runs/segment/treinamentos/res_1408/weights/last.pt",
                     imgsz_load=1280, imgsz_person=1600),
    "control_1280": dict(pesos="runs/segment/treinamentos/control_1280/weights/last.pt",
                         imgsz_load=1280, imgsz_person=1792),
}


def caminhos_val(data_yaml: str) -> list[tuple[Path, Path]]:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    raiz = Path(cfg.get("path", Path(data_yaml).parent))
    imgs = raiz / cfg["val"]
    lbls = raiz / str(cfg["val"]).replace("images", "labels", 1)
    return [(p, lbls / f"{p.stem}.txt") for p in sorted(imgs.glob("*.jpg"))]


def ler_gt(lbl: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    if not lbl.exists():
        return []
    out = []
    for linha in lbl.read_text().splitlines():
        p = linha.split()
        if len(p) < 7:
            continue
        pts = np.array([float(v) for v in p[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        out.append((int(p[0]), pts.astype(np.int32)))
    return out


def escolher(pares: list[tuple[Path, Path]], n: int, lacuna: int = 600) -> list[tuple[Path, Path]]:
    """Cobre os 4 dias, prioriza imagens com mais pessoas, garante 1 background.

    Nunca escolhe dois quadros a menos de `lacuna` segundos um do outro: a camera
    e fixa e imagens proximas no tempo sao praticamente identicas, o que
    desperdicaria metade dos exemplos mostrando a mesma cena.
    """
    from datetime import datetime

    def ts(nome: str) -> datetime:
        return datetime.strptime(nome.split("bullet_")[1][:15], "%Y%m%d_%H%M%S")

    por_dia = defaultdict(list)
    for img, lbl in pares:
        gt = ler_gt(lbl, 1, 1)
        por_dia[ts(img.name).date()].append((sum(c == 1 for c, _ in gt), len(gt), img, lbl))
    for dia in por_dia:
        por_dia[dia].sort(key=lambda t: (-t[0], -t[1]))

    escolhidas: list = []

    def cabe(cand) -> bool:
        return all(abs((ts(cand[2].name) - ts(e[2].name)).total_seconds()) >= lacuna
                   for e in escolhidas)

    dias = sorted(por_dia)
    for _ in range(max(len(v) for v in por_dia.values())):
        for dia in dias:
            if len(escolhidas) >= n:
                break
            for cand in por_dia[dia]:
                if cand not in escolhidas and cabe(cand):
                    escolhidas.append(cand)
                    break
        if len(escolhidas) >= n:
            break

    if not any(t[1] == 0 for t in escolhidas):  # garante um background
        vazias = [t for d in dias for t in por_dia[d] if t[1] == 0]
        if vazias:
            escolhidas[-1] = vazias[0]
    return [(t[2], t[3]) for t in escolhidas]


def prever(modelo: YOLO, img: Path, imgsz: int, classes: list[int], conf: float):
    r = modelo.predict(str(img), imgsz=imgsz, classes=classes, conf=conf,
                       rect=True, verbose=False, retina_masks=True)[0]
    saida = []
    if r.masks is not None:
        for poly, c, cf in zip(r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            saida.append((int(c), np.asarray(poly, dtype=np.int32), float(cf)))
    return saida


def desenhar(img: Path, pred: list) -> np.ndarray:
    """Desenha somente o que a rede detectou: mascara, contorno, classe e confianca."""
    base = cv2.imread(str(img))
    capa = base.copy()
    for c, poly, _ in pred:
        cv2.fillPoly(capa, [poly], CORES.get(c, BRANCO))
    base = cv2.addWeighted(capa, 0.35, base, 0.65, 0)

    for c, poly, cf in pred:
        cor = CORES.get(c, BRANCO)
        cv2.polylines(base, [poly], True, cor, 2, cv2.LINE_AA)
        txt = f"{NOMES.get(c, c)} {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = int(poly[:, 0].min())
        y = max(int(poly[:, 1].min()), th + 5)
        cv2.rectangle(base, (x, y - th - 4), (x + tw + 4, y), cor, -1)
        cv2.putText(base, txt, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="yolo_dataset/data.yaml")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="examples")
    args = ap.parse_args()

    pares = caminhos_val(args.data)
    sel = escolher(pares, args.n)
    print(f"\n  {len(sel)} imagens de validacao escolhidas de {len(pares)}\n")

    for nome, cfg in MODELOS.items():
        pesos = Path(cfg["pesos"])
        if not pesos.exists():
            print(f"  pulando {nome}: {pesos} nao existe")
            continue
        destino = Path(args.out) / nome
        destino.mkdir(parents=True, exist_ok=True)
        modelo = YOLO(str(pesos))
        totais = defaultdict(lambda: [0, 0])
        linhas = []
        print(f"  {nome}  (load @{cfg['imgsz_load']}, person @{cfg['imgsz_person']})")
        for img, lbl in sel:
            base = cv2.imread(str(img))
            gt = ler_gt(lbl, base.shape[1], base.shape[0])
            pred = (prever(modelo, img, cfg["imgsz_load"], [0, 2], args.conf)
                    + prever(modelo, img, cfg["imgsz_person"], [1], args.conf))
            cv2.imwrite(str(destino / img.name), desenhar(img, pred),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            for c, _ in gt:
                totais[c][1] += 1
            for c, _, _ in pred:
                totais[c][0] += 1
            det = defaultdict(int)
            for c, _, _ in pred:
                det[c] += 1
            an = defaultdict(int)
            for c, _ in gt:
                an[c] += 1
            linhas.append(f"{img.name}  " + "  ".join(
                f"{NOMES[c]} {det[c]}/{an[c]}" for c in sorted(set(an) | set(det))) or f"{img.name}  (vazia)")
            print(f"    {img.name}")
        resumo = (f"modelo: {pesos}\nload @{cfg['imgsz_load']}  person @{cfg['imgsz_person']}  "
                  f"conf={args.conf}\n\ndetectado/anotado por imagem:\n" + "\n".join(linhas)
                  + "\n\ntotal:\n" + "\n".join(
                      f"  {NOMES[c]}: detectou {v[0]}, anotadas {v[1]}" for c, v in sorted(totais.items())) + "\n")
        (destino / "resumo.txt").write_text(resumo)
        print(f"    -> {destino}/  ({len(sel)} imagens + resumo.txt)\n")


if __name__ == "__main__":
    main()
