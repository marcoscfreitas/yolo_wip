"""Avaliacao por classe do YOLO-seg iCrane.

  python evaluate.py runs/segment/treinamentos/res_1408-2          # todos os checkpoints da run
  python evaluate.py a/best.pt b/best.pt                          # compara runs
  python evaluate.py a/best.pt --sweep                            # varias resolucoes

Por que este script existe em vez de `yolo val`:

O "all" que o Ultralytics imprime e a media simples das 3 classes, e `pipe` tem
apenas 3 instancias na validacao (contra 733 de `load` e 109 de `person`). Ou
seja, 1/3 do numero principal vem de 3 objetos. O salto de mAP50 0.47 -> 0.67
entre as epocas 63 e 65 da run treino_1280_yolo26 foi so o `pipe` acendendo, nao
o modelo melhorando. Pior: o fitness que escolhe o best.pt e
mAP50-95(mask) + mAP50-95(box) sobre a media das classes (utils/metrics.py:1369),
entao esses 3 objetos tambem escolhem qual checkpoint e salvo.

Este script reporta sempre por classe e calcula um agregado que ignora classes
com menos de --min-inst instancias, para o numero de cima ser comparavel entre
runs.
"""

import argparse
import csv as _csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

SWEEP = [1280, 1408, 1600, 1792]
REGISTRO = Path("resultados/avaliacoes.csv")
COLUNAS = ["quando", "checkpoint", "imgsz", "split", "classe", "inst",
           "P", "R", "mAP50", "mAP50-95", "mAP50(M)", "mAP50-95(M)"]


def registrar(linhas: list[dict]) -> None:
    """Grava as metricas em CSV para poderem ser lidas sem copiar do terminal."""
    REGISTRO.parent.mkdir(exist_ok=True)
    novo = not REGISTRO.exists()
    with REGISTRO.open("a", newline="") as f:
        w = _csv.DictWriter(f, COLUNAS)
        if novo:
            w.writeheader()
        w.writerows(linhas)


def contar_instancias(data_yaml: str, split: str) -> Counter:
    """Conta instancias por classe lendo as labels direto do disco."""
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    raiz = Path(cfg.get("path", Path(data_yaml).parent))
    imgs = Path(cfg[split])
    labels = raiz / str(imgs).replace("images", "labels", 1)
    n = Counter()
    for txt in labels.glob("*.txt"):
        for linha in txt.read_text().splitlines():
            if linha.strip():
                n[int(linha.split()[0])] += 1
    return n


def avaliar(pesos: str, data: str, imgsz: int, split: str, rect: bool):
    modelo = YOLO(pesos)
    return modelo.val(
        data=data, imgsz=imgsz, split=split, rect=rect,
        batch=1, plots=False, verbose=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pesos", nargs="+",
                    help="arquivos .pt e/ou pastas de run (expande weights/*.pt)")
    ap.add_argument("--data", default="yolo_dataset/data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--sweep", action="store_true",
                    help=f"avalia em varias resolucoes: {SWEEP}")
    ap.add_argument("--no-rect", action="store_true")
    ap.add_argument("--min-inst", type=int, default=10,
                    help="classes abaixo disso saem do agregado (default: 10)")
    args = ap.parse_args()

    n_inst = contar_instancias(args.data, args.split)
    resolucoes = SWEEP if args.sweep else [args.imgsz]
    ranking = []

    pesos = []
    for alvo in args.pesos:
        a = Path(alvo)
        if a.is_dir():
            achados = sorted((a / "weights").glob("*.pt")) or sorted(a.glob("*.pt"))
            if not achados:
                print(f"  aviso: nenhum .pt em {a}")
            pesos += achados
        else:
            pesos.append(a)

    for peso, imgsz in ((p, i) for p in pesos for i in resolucoes):
        r = avaliar(str(peso), args.data, imgsz, args.split, not args.no_rect)
        idx = list(r.ap_class_index)

        rotulo = f"{peso.parent.parent.name}/{peso.name}" if peso.parent.name == "weights" else str(peso)
        print(f"\n{'=' * 78}\n  {rotulo}   split={args.split}   imgsz={imgsz}\n{'=' * 78}")
        print(f"  {'classe':<10}{'inst':>7}{'P(B)':>9}{'R(B)':>9}"
              f"{'mAP50(B)':>11}{'mAP50-95(B)':>13}{'mAP50(M)':>11}{'mAP50-95(M)':>13}")

        confiaveis, linhas = [], []
        agora = f"{datetime.now():%Y-%m-%d %H:%M}"
        for i, c in enumerate(idx):
            pb, rb, ap50b, apb, _pm, _rm, ap50m, apm = r.class_result(i)
            nome, n = r.names[c], n_inst.get(c, 0)
            flag = "" if n >= args.min_inst else "   <- ruido, fora do agregado"
            print(f"  {nome:<10}{n:>7}{pb:>9.3f}{rb:>9.3f}"
                  f"{ap50b:>11.3f}{apb:>13.3f}{ap50m:>11.3f}{apm:>13.3f}{flag}")
            linhas.append(dict(zip(COLUNAS, [agora, rotulo, imgsz, args.split, nome, n,
                                             f"{pb:.4f}", f"{rb:.4f}", f"{ap50b:.4f}",
                                             f"{apb:.4f}", f"{ap50m:.4f}", f"{apm:.4f}"])))
            if n >= args.min_inst:
                confiaveis.append((ap50b, apb, ap50m, apm))

        media = r.mean_results()  # [P, R, mAP50, mAP50-95] box + idem mask
        print(f"  {'-' * 76}")
        print(f"  {'all (ultra)':<10}{sum(n_inst.values()):>7}{media[0]:>9.3f}{media[1]:>9.3f}"
              f"{media[2]:>11.3f}{media[3]:>13.3f}{media[6]:>11.3f}{media[7]:>13.3f}")
        if confiaveis:
            m = [sum(col) / len(confiaveis) for col in zip(*confiaveis)]
            nota = ("   <- use este para comparar runs" if len(confiaveis) < len(idx)
                    else "   <- nenhuma classe de ruido; igual ao 'all'")
            print(f"  {'AGREGADO':<10}{len(confiaveis):>7}{'':>9}{'':>9}"
                  f"{m[0]:>11.3f}{m[1]:>13.3f}{m[2]:>11.3f}{m[3]:>13.3f}{nota}")
            ranking.append((m[0], m[1], m[2], rotulo, imgsz))
            linhas.append(dict(zip(COLUNAS, [agora, rotulo, imgsz, args.split, "AGREGADO", "",
                                             "", "", f"{m[0]:.4f}", f"{m[1]:.4f}",
                                             f"{m[2]:.4f}", f"{m[3]:.4f}"])))
        registrar(linhas)

    if len(ranking) > 1:
        print(f"\n{'=' * 78}\n  RANKING pelo AGREGADO (sem as classes de ruido)\n{'=' * 78}")
        print(f"  {'mAP50':>8}{'mAP50-95':>11}{'mAP50(M)':>11}  {'checkpoint':<40}{'imgsz':>7}")
        for ap50, ap, ap50m, rotulo, imgsz in sorted(ranking, reverse=True):
            print(f"  {ap50:>8.3f}{ap:>11.3f}{ap50m:>11.3f}  {rotulo:<40}{imgsz:>7}")
        print()

    if args.sweep:
        print("  Nota: o imgsz de inferencia nao precisa ser igual ao de treino, e o\n"
              "  melhor valor difere por modelo. Escolha pelo RANKING acima.\n")


if __name__ == "__main__":
    main()
