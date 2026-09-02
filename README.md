# iCrane — segmentação de cargas e pessoas

Modelo YOLO de segmentação para uma câmera fixa na ponta de um guindaste
embarcado, identificando **cargas**, **pessoas** e **tubos** durante as
operações.

> **As imagens não estão neste repositório.** O dataset, os vídeos, as
> anotações e as saídas de inferência são dados sensíveis da empresa e estão
> excluídos pelo `.gitignore`. Aqui vai apenas o código, a documentação e a
> configuração.

## Documentação

**[docs/NOTAS_TECNICAS.md](docs/NOTAS_TECNICAS.md)** — achados verificados com os números
que os sustentam: limites de VRAM, armadilhas do Ultralytics já identificadas no
código-fonte, características do dataset, resultados medidos e piso de ruído.
**Ler antes de mexer nos hiperparâmetros.**

## Scripts

| script | o que faz |
|---|---|
| `train.py` | treina uma variante (`control`, `res`, `mosaic`); valida o ambiente e o orçamento de VRAM antes de começar |
| `evaluate.py` | avalia checkpoints **por classe**, varre resoluções de inferência e ranqueia; grava em `resultados/avaliacoes.csv` |
| `resplit.py` | refaz o split treino/validação por bloco temporal, sem vazamento entre quadros vizinhos |
| `examples.py` | desenha as detecções sobre imagens de validação, para inspeção visual |

Em [extras/](extras/) ficam experimentos fora do pipeline (fluxo óptico).

## Uso

Sempre pelo caminho absoluto do interpretador — existe um ultralytics antigo em
`~/.local` que o python do sistema enxerga, e ele não tem `cls_remap` (vale ~13
pontos de mAP50 em `person`). Os scripts abortam se isso acontecer.

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

## Ambiente

- AMD Radeon RX 7600, 8 GB — **também é a GPU de vídeo do desktop**, então o
  orçamento real de treino é ~5 GB. `train.py` recusa configurações acima disso
  a menos que se passe `--gpu-livre` com a sessão gráfica parada.
- Ultralytics ≥ 8.4.121, torch 2.13 + ROCm 7.2, no `venv/` do projeto.
- 15 GB de RAM divididos com a IDE.

Os detalhes e os números por trás desses limites estão em
[docs/NOTAS_TECNICAS.md](docs/NOTAS_TECNICAS.md).
