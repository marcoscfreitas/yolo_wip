"""Treino YOLO-seg iCrane.

    ./venv/bin/python train.py            # variante `res` (a atual)
    ./venv/bin/python train.py control    # baseline de comparacao
    ./venv/bin/python train.py res --imgsz 1600 --gpu-livre

Sempre chamar `./venv/bin/python` por caminho absoluto: existe um ultralytics
antigo em ~/.local que o python do sistema enxerga, e ele nao tem `cls_remap`
(vale ~13 pontos de mAP50 em `person`). O script aborta se isso acontecer.

Rodar de um `tmux` num terminal comum, nao do terminal integrado da IDE: la o
processo herda o escopo do Electron e vira alvo do OOM killer, derrubando a IDE.

VRAM: a placa tambem desenha o desktop, entao o orcamento e ~5 GB, nao 8. Para
usar os 8 GB, parar a sessao grafica e passar --gpu-livre:

    sudo systemctl stop gdm   # Ctrl+Alt+F3 antes
    ./venv/bin/python train.py res --imgsz 1600 --gpu-livre
    sudo systemctl start gdm

O porque de cada escolha de hiperparametro, os numeros que a sustentam e as
armadilhas do Ultralytics ja verificadas estao em docs/NOTAS_TECNICAS.md. Ler antes
de mexer -- varias delas parecem inofensivas e nao sao.
"""

import argparse
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

# Base comum. Os comentarios abaixo marcam os parametros que JA causaram
# regressao quando alterados; o resto e default do Ultralytics.
BASE = dict(
    data="yolo_dataset/data.yaml",
    epochs=100,        # onde as perdas de validacao ainda estavam caindo; 150 overfitou
    patience=0,        # desligado: o fitness e a media das classes e oscila por ruido
    batch=4,
    rect=True,
    workers=2,
    device=0,
    project="treinamentos",
    cos_lr=True,       # define o FORMATO do cosseno, nao so o ponto de parada
    cache=False,       # 'disk' deixa .npy stale; 'ram' custa 700 MB por ~0.16 s/epoca
    save_period=10,    # checkpoints periodicos, para o evaluate.py ranquear
    # --- augmentation ---
    copy_paste=0.3,
    copy_paste_mode="flip",
    degrees=10.0,      # NAO remover: e o que segura o overfitting com ~124 imagens
    shear=2.0,         # idem
    mixup=0.0,         # inerte com rect=True; explicito para nao enganar
    cutmix=0.0,
    mosaic=0.0,        # idem: rect=True zera mosaic (dataset.py:313)
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

# Cada variante muda UMA coisa em relacao ao control. Manter assim: sem isso
# nao da para atribuir efeito, e o piso de ruido do `person` e de +-0.04.
VARIANTES = {
    "control": dict(imgsz=1280),
    "res": dict(imgsz=1408),                       # melhor em `load`: +0.075 mAP50
    "mosaic": dict(imgsz=1024, rect=False, mosaic=1.0, close_mosaic=30),
}


def vram_estimada(imgsz: int, rect: bool, batch: int) -> float:
    """Ajuste linear sobre picos medidos; erro < 0.02 GB. Ver docs/NOTAS_TECNICAS.md."""
    altura = round(imgsz * 9 / 16 / 32) * 32 if rect else imgsz
    return (5.20 * (imgsz * altura / 1e6) - 1.04) * batch / 4


def checar_ambiente() -> None:
    from ultralytics.cfg import DEFAULT_CFG_DICT

    ok = "cls_remap" in DEFAULT_CFG_DICT
    info = (f"python      : {sys.executable}\n"
            f"ultralytics : {ultralytics.__version__}\n"
            f"cls_remap   : {'OK' if ok else 'AUSENTE'}\n")
    print("\n" + info)

    log = Path("resultados/ambiente.log")
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as f:
        from datetime import datetime
        f.write(f"--- {datetime.now():%Y-%m-%d %H:%M}\n{info}")

    if not ok:
        sys.exit(
            "  ERRO: ultralytics sem cls_remap. A classe `person` nasceria do zero\n"
            "  em vez de herdar o cabecalho de `person` do COCO (~13 pontos de mAP50).\n"
            "  Rode por caminho absoluto:  ./venv/bin/python train.py\n")

    if npys := list(Path("yolo_dataset").rglob("*.npy")):
        print(f"  aviso: {len(npys)} .npy stale de cache='disk' antigo.\n"
              f"  limpe com:  find yolo_dataset -name '*.npy' -delete\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variante", nargs="?", default="res", choices=list(VARIANTES))
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--gpu-livre", action="store_true",
                    help="dispensa o teto de ~5 GB; use so com a sessao grafica parada")
    for k in ("epochs", "batch", "imgsz"):
        ap.add_argument(f"--{k}", type=int)
    ap.add_argument("--name")
    args = ap.parse_args()

    checar_ambiente()

    cfg = {**BASE, **VARIANTES[args.variante]}
    for k in ("epochs", "batch", "imgsz"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg["name"] = args.name or f"{args.variante}_{cfg['imgsz']}"

    gb = vram_estimada(cfg["imgsz"], cfg["rect"], cfg["batch"])
    print(f"  {args.variante} -> treinamentos/{cfg['name']}")
    print(f"  imgsz={cfg['imgsz']} batch={cfg['batch']} rect={cfg['rect']} "
          f"epochs={cfg['epochs']}  |  VRAM estimada ~{gb:.1f} GB")
    if gb > 5.2 and not args.gpu_livre:
        sys.exit("\n  ERRO: acima do orcamento de ~5 GB para GPU compartilhada com o\n"
                 "  desktop. Isso trava a sessao grafica. Reduza o imgsz/batch, ou pare\n"
                 "  o gdm e passe --gpu-livre.\n")
    print()

    YOLO(args.model).train(**cfg)

    print(f"\n  ./venv/bin/python evaluate.py runs/segment/treinamentos/{cfg['name']}\n")


if __name__ == "__main__":
    main()
