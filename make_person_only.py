"""Gera um .pt generico do COCO que detecta SOMENTE `person`.

  ./venv/bin/python make_person_only.py                      # yolo26n-seg.pt -> person_only.pt
  ./venv/bin/python make_person_only.py --modelo yolo26n.pt  # versao so de deteccao
  ./venv/bin/python make_person_only.py --classe 0 --nome person

POR QUE ESTE SCRIPT EXISTE

Filtrar com `model.predict(..., classes=[0])` funciona, mas o filtro vive no
codigo de quem chama, nao no arquivo. E nao ha como gravar isso no .pt:

  - `m.overrides["classes"] = [0]`  funciona em memoria, mas o `save()` descarta.
  - `m.model.args["classes"] = [0]` idem: o `save()` filtra os args por uma
    whitelist, e a chave some.

Verificado nas duas formas: o .pt recarregado volta a detectar as 80 classes.

Entao, para entregar um arquivo unico que nao dependa de ninguem lembrar do
argumento, o jeito e cortar a head: manter so a linha da classe desejada nas
ultimas convolucoes dos ramos de classificacao, e declarar nc=1. A rede fica
literalmente incapaz de emitir outra classe.

O `yolo26n-seg` e end2end e tem DOIS ramos de classificacao (`cv3` para o
treino one2many e `one2one_cv3` para a inferencia), com pesos distintos, em 3
escalas cada. Os seis precisam ser cortados; cortar so um deixa o modelo
inconsistente entre treino e inferencia.

O script so aceita o resultado se as deteccoes de `person` sairem identicas as
do modelo original (mesmas caixas, mesmas confiancas, mesmas mascaras).
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

AMOSTRAS = ["bus.jpg", "zidane.jpg"]


def ultima_conv(seq: nn.Module) -> nn.Conv2d:
    return [m for m in seq.modules() if isinstance(m, nn.Conv2d)][-1]


def cortar(conv: nn.Conv2d, idx: int) -> None:
    """Deixa a conv com um unico canal de saida: o da classe `idx`."""
    with torch.no_grad():
        w = conv.weight.data[idx : idx + 1].clone()
        b = conv.bias.data[idx : idx + 1].clone() if conv.bias is not None else None
    conv.out_channels = 1
    conv.weight = nn.Parameter(w, requires_grad=conv.weight.requires_grad)
    if b is not None:
        conv.bias = nn.Parameter(b, requires_grad=conv.bias.requires_grad)


def deteccoes(modelo: YOLO, imgs: list[str], classe: int):
    """Caixas/confiancas/mascaras apenas da classe alvo, para comparar antes x depois."""
    out = []
    for img in imgs:
        r = modelo.predict(img, conf=0.001, verbose=False)[0]
        sel = (r.boxes.cls.int() == classe).nonzero().view(-1)
        m = r.masks.data[sel].cpu() if r.masks is not None else None
        out.append((r.boxes.xyxy[sel].cpu(), r.boxes.conf[sel].cpu(), m))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="yolo26n-seg.pt")
    ap.add_argument("--classe", type=int, default=0, help="indice COCO (person = 0)")
    ap.add_argument("--nome", default="person")
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    assets = Path(YOLO.__module__ and __import__("ultralytics").__file__).parent / "assets"
    imgs = [str(assets / n) for n in AMOSTRAS if (assets / n).exists()]
    if not imgs:
        raise SystemExit("  amostras do ultralytics nao encontradas; sem elas nao da para verificar")

    original = YOLO(args.modelo)
    if args.classe not in original.model.names:
        raise SystemExit(f"  classe {args.classe} nao existe em {args.modelo}")
    nome_orig = original.model.names[args.classe]
    print(f"  {args.modelo}: {len(original.model.names)} classes, "
          f"indice {args.classe} = '{nome_orig}'")
    antes = deteccoes(original, imgs, args.classe)

    m = YOLO(args.modelo)
    head = m.model.model[-1]
    extra = head.no - head.nc  # canais de regressao, preservados

    cortadas = 0
    for ramo in ("cv3", "one2one_cv3"):
        br = getattr(head, ramo, None)
        if br is None:
            continue
        for seq in br:
            cortar(ultima_conv(seq), args.classe)
            cortadas += 1
    if cortadas == 0:
        raise SystemExit("  nenhum ramo de classificacao encontrado; head inesperada")

    head.nc = 1
    head.no = 1 + extra
    m.model.nc = 1
    m.model.names = {0: args.nome}
    if isinstance(getattr(m.model, "yaml", None), dict):
        m.model.yaml["nc"] = 1
    m.model.model[-1] = head
    print(f"  cortadas {cortadas} convolucoes de classificacao; nc: 80 -> {head.nc}, "
          f"no: {head.nc + extra}")

    saida = Path(args.saida or f"{Path(args.modelo).stem}-{args.nome}.pt")
    m.save(str(saida))

    # --- verificacao: o arquivo gravado, recarregado do zero ---
    novo = YOLO(str(saida))
    print(f"\n  {saida} ({saida.stat().st_size / 1e6:.1f} MB)")
    print(f"  classes do arquivo recarregado: {novo.model.names}")
    depois = deteccoes(novo, imgs, 0)

    ok = True
    for img, (b0, c0, m0), (b1, c1, m1) in zip(imgs, antes, depois):
        nome = Path(img).name
        if b0.shape != b1.shape:
            print(f"    {nome}: FALHOU - {b0.shape[0]} vs {b1.shape[0]} deteccoes")
            ok = False
            continue
        db = (b0 - b1).abs().max().item() if b0.numel() else 0.0
        dc = (c0 - c1).abs().max().item() if c0.numel() else 0.0
        dm = (m0 - m1).abs().max().item() if m0 is not None and m1 is not None and m0.numel() else 0.0
        bom = db < 1e-3 and dc < 1e-4 and dm < 1e-3
        ok &= bom
        print(f"    {nome}: {b0.shape[0]} deteccoes de '{args.nome}'  "
              f"dif max  caixa={db:.2e}  conf={dc:.2e}  mascara={dm:.2e}  "
              f"{'OK' if bom else 'FALHOU'}")

    print("\n  " + ("identico ao original na classe alvo; pode embarcar"
                    if ok else "DIVERGIU - nao usar este arquivo"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
