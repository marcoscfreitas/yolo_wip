# iCrane — segmentação de cargas e pessoas

Modelo YOLO de segmentação para uma câmera fixa na ponta de um guindaste
embarcado, identificando **cargas**, **pessoas** e **tubos** durante as
operações.

## Documentação

**[docs/NOTAS_TECNICAS.md](docs/NOTAS_TECNICAS.md)** — achados verificados com os
números que os sustentam: limites de VRAM, armadilhas do Ultralytics,
características do dataset, resultados medidos e piso de ruído.
**Ler antes de mexer nos hiperparâmetros.**

## Scripts

| script | o que faz |
|---|---|
| `train.py` | treina uma variante (`control`, `res`, `mosaic`); valida o ambiente e o orçamento de VRAM antes de começar |
| `evaluate.py` | avalia checkpoints **por classe**, varre resoluções de inferência e ranqueia; grava em `resultados/avaliacoes.csv` |
| `resplit.py` | refaz o split treino/validação por bloco temporal, sem vazamento entre quadros vizinhos |
| `examples.py` | desenha as detecções sobre imagens de validação, para inspeção visual |

Em [optical_flow/](optical_flow/) ficam experimentos fora do pipeline (fluxo óptico).

## Uso

Sempre pelo caminho absoluto do interpretador do `venv/` do projeto — os scripts
abortam se pegarem outro ultralytics.

```bash
tmux new -s treino                        # nao usar o terminal integrado da IDE
./venv/bin/python train.py                # variante `res` (1408)
./venv/bin/python evaluate.py runs/segment/treinamentos/res_1408
./venv/bin/python examples.py
```

Para refazer o split (reversível com `--reverter`):

```bash
./venv/bin/python resplit.py --dry-run --limiar 600
./venv/bin/python resplit.py --limiar 600
```
