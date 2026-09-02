# Notas técnicas — iCrane YOLO-seg

Achados verificados, com os números que os sustentam. Serve para não repetir
experimento já feito nem reintroduzir bug já corrigido. A seção "Correções" no
fim lista as conclusões do relatório de experimentos anterior que estas medições
desmentiram.

---

## 1. Limites de hardware

A **RX 7600 (8 GB) também desenha o desktop.** GNOME, navegador e a IDE
consomem VRAM ao mesmo tempo, então sobram **~5 GB para o treino, não 8**.
Passar disso trava a sessão gráfica: o kernel registra
`[drm] Failed to pin framebuffer with error -12` e
`amdgpu: Not enough memory for command submission!`, e o GNOME Shell reinicia.

Picos de VRAM medidos (yolo26n-seg, batch=4):

| config | pixels de entrada | pico |
|---|---|---|
| `rect=True` 1280 | 0,94 Mpx | 3,86 GB |
| `rect=True` 1408 | 1,12 Mpx | 4,82 GB |
| `rect=True` 1600 | 1,43 Mpx | 6,42 GB |
| `rect=False` 1280 | 1,64 Mpx | 7,49 GB |

Modelo linear que reproduz os três pontos medidos com erro < 0,02 GB:

    VRAM ≈ 5,20 × Mpx_entrada − 1,04     (batch = 4)

Para usar os 8 GB de verdade, parar a sessão gráfica e treinar de um TTY
(`Ctrl+Alt+F3`, `sudo systemctl stop gdm`). Libera a VRAM do desktop e ~3 GB de RAM.

**RAM do host:** 15 GB divididos com a IDE. Não lançar treino pelo terminal
integrado da IDE — o processo herda o escopo systemd do Electron e
`oom_score_adj=100`, virando alvo preferencial do OOM killer; quando morre, leva
a IDE junto. Usar `tmux` a partir de um terminal comum.

**`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` não usar:** causou
`Memory access fault by GPU node-1` neste stack ROCm.

---

## 2. Armadilhas do Ultralytics verificadas no código-fonte

### `rect=True` zera mosaic, mixup e cutmix

`ultralytics/data/dataset.py:313-315`:

```python
hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
hyp.mixup  = hyp.mixup  if self.augment and not self.rect else 0.0
hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
```

Todas as runs deste projeto usam `rect=True`, então **nunca houve mosaic nem
mixup**, apesar de `mosaic=1.0` e `mixup=0.15` na config. `close_mosaic` é
no-op. Confirmado visualmente nos `train_batch*.jpg`: são imagens inteiras, não
mosaicos.

O shuffle **não** é afetado aqui: `detect/train.py:94` só desliga o shuffle
quando os batches têm shapes diferentes, e todas as imagens são 1920×1080.

### `cls_remap` vale ~13 pontos de mAP50 em `person`

O `yolo26n-seg.pt` é pré-treinado no COCO (80 classes, inclui `person`, não
inclui `load` nem `pipe`). O Ultralytics ≥ 8.4.12x copia a linha do cls head da
classe pré-treinada que casa por nome — o log imprime
`Remapped 1/3 cls head rows from pretrained weights by class name`. Versões
8.4.66/8.4.67 **não têm** esse recurso.

Mesma receita, mesmo split, avaliado em 1600:

| | load mAP50 | person mAP50 |
|---|---|---|
| 8.4.121 (com `cls_remap`) | 0,671 | **0,463** |
| 8.4.66 (sem) | 0,660 | 0,336 |

`load` e `pipe` não mudam — nenhum nome do COCO casa com eles. O `train.py`
aborta se rodar numa versão sem `cls_remap`.

O 8.4.66 que estava instalado em `~/.local` foi desinstalado; a checagem de
`cls_remap` fica como rede de segurança caso um `pip install --user` o traga de
volta. Chamar sempre `./venv/bin/python` pelo caminho absoluto: `source
venv/bin/activate` **não** resolve dentro de uma sessão tmux criada antes da
ativação, porque ela herda o ambiente do servidor tmux.

### `patience` e `best.pt` são alimentados pela média das classes

O fitness é `mAP50-95(mask) + mAP50-95(box)` sobre a média das classes
(`utils/metrics.py:1369`). Com uma classe de pouquíssimas instâncias, ela
acende e apaga sozinha e move a média inteira. Quando `pipe` tinha 3 instâncias
na validação, valia ±0,33 de mAP50 — mais que qualquer efeito real que
estávamos tentando medir, e disparava o early stopping por ruído. Daí
`patience=0`.

### `cache="disk"` deixa `.npy` órfãos

Ficam ao lado das imagens e viram cache stale se um arquivo for substituído
mantendo o nome. Havia 206 órfãos no dataset. Usar `cache=False`: o disco lê a
~540 MB/s com imagens de 758 KB, então o cache economizava ~0,16 s numa época
de 24 s e custava ~700 MB de RAM.

---

## 3. Os dados

162 imagens 1920×1080 de uma câmera bullet fixa na ponta do guindaste, em 4
dias. Classes: `load`, `person`, `pipe`.

**Tamanho dos objetos** (raiz da área do bbox, em pixels na imagem original):

| classe | treino | validação | % abaixo de 32 px (val) |
|---|---|---|---|
| load | mediana 118 px | mediana 82 px | 0% |
| person | mediana 33 px | **mediana 20 px** | **81%** |
| pipe | mediana 263 px | — | — |

Em `imgsz=1280` a imagem encolhe 1080→720, então uma pessoa mediana chega à
rede com **13 px**. O head P3 tem stride 8. É o teto físico do `person`.

**`pipe` é inutilizável como métrica:** 45 instâncias no treino, **0 na
validação** depois do resplit. Continua sendo treinado, mas não é medido — o que
tem o efeito colateral bom de limpar o fitness e acabar com a loteria do
`best.pt`.

### Split por bloco temporal (`resplit.py`)

Sortear imagens individualmente vaza: a lacuna mínima entre quadros
consecutivos é de **1 s**, a mediana 20 s, e 85 dos 158 pares consecutivos estão
abaixo de 30 s. São praticamente duplicatas.

O split antigo tinha o problema oposto — separava dias inteiros, e o treino
nunca via 2 das 4 condições em que era avaliado.

O `resplit.py` agrupa em sessões por lacuna temporal e distribui **sessões
inteiras**, garantindo que todos os dias apareçam nos dois lados. Estado atual
(`--limiar 600`):

| | imagens | dias | load | person | pipe | background |
|---|---|---|---|---|---|---|
| treino | 124 | todos os 4 | 1721 | 553 | 45 | 12 (10%) |
| validação | 38 | todos os 4 | 627 | 111 | 0 | 9 (24%) |

**Menor distância temporal entre uma imagem de treino e uma de validação no
mesmo dia: 877 s.** Esse número é a garantia de que não houve vazamento.
Reversível com `resplit.py --reverter`.

---

## 4. Resultados medidos no split atual

Média dos 5 checkpoints tardios (ep 60, 70, 80, 90, last), avaliados em 1600,
com a faixa min–max:

| | control (1280) | res (1408) | |
|---|---|---|---|
| load mAP50 | 0,536 [0,522–0,555] | **0,610** [0,600–0,622] | **efeito real** |
| load mAP50-95 | 0,446 [0,433–0,460] | **0,505** [0,488–0,520] | **efeito real** |
| load máscara | 0,536 [0,521–0,555] | **0,607** [0,601–0,619] | **efeito real** |
| person mAP50 | 0,446 [0,393–0,470] | 0,398 [0,383–0,428] | faixas se sobrepõem |
| person mAP50-95 | 0,253 [0,235–0,265] | 0,229 [0,221–0,248] | faixas se sobrepõem |

**Piso de ruído medido:** `load` ±0,015, `person` **±0,04**. São 111 instâncias
de `person` em ~29 imagens. Qualquer experimento precisa mover o `person` mais
de 8 pontos para ser detectável — todas as discussões anteriores sobre `person`
estavam dentro do ruído.

### A resolução de inferência preferida difere por classe

Sweep do `last.pt` de cada modelo:

| modelo | imgsz | load mAP50 | person mAP50 | person máscara |
|---|---|---|---|---|
| control_1280 | **1280** | 0,560 | 0,465 | 0,387 |
| control_1280 | 1600 | 0,535 | 0,467 | 0,465 |
| control_1280 | **1792** | 0,539 | **0,482** | **0,467** |
| res_1408 | **1280** | **0,644** | 0,383 | 0,344 |
| res_1408 | 1600 | 0,610 | 0,395 | 0,405 |
| res_1408 | 1792 | 0,611 | 0,333 | 0,337 |

`load` é melhor em **1280 nos dois modelos**; `person` na resolução mais alta,
**nos dois**. Faz sentido: cargas são grandes e ampliar demais as tira da escala
de treino; pessoas têm 20 px e ampliar as traz para a faixa detectável.

Como o mAP é por classe, **dois passes em resoluções diferentes com roteamento
por classe** entregam exatamente os melhores números da tabela. É o que o
`examples.py` faz. A inferência é 14 ms, então dois passes custam 28 ms.

---

## 5. O que não funcionou

- **Remover `degrees`/`shear`** achando que câmera fixa não gira. Com 100
  imagens, a augmentation geométrica não está ali para imitar a realidade, está
  para impedir a rede de decorar. Removida a rotação e alongado o treino para
  150 épocas, as três perdas de validação viraram para cima a partir da época
  ~76 enquanto a de treino despencava para 0,43. Overfitting claro.
- **`epochs=300`** a 1408: 2 h de treino sem ganho, e `cos_lr` faz o `epochs`
  definir o formato do cosseno, não só o ponto de parada.
- **`patience` menor que o schedule** com `cos_lr`: mata a run antes do anneal.
- **Inferir sempre em 1600**: valia no split antigo. No atual é a pior escolha
  para `load` nos dois modelos.

---

## 6. Correções ao `relatorio_experimentos.md`

- O "Sweet Spot" de 1280 com `batch=4` não era o máximo viável: usava 3,86 dos
  8 GB. O limite real é o desktop compartilhando a placa, não a placa.
- `batch=2` não "quebra a matemática do gradiente" — o Ultralytics acumula até
  `nbs=64` (`engine/trainer.py:297`). O problema de batch pequeno é a
  estatística de BatchNorm.
- `mixup` nunca criou "cargas fantasmas": estava desligado por `rect=True`.
  `copy_paste` estava ativo e não gera fantasma (cola instância + máscara +
  label).
- `pipe` com mAP50 de 99,5% eram 3 objetos. Não era sinal.
- A conclusão de que modelos maiores que o Nano são inviáveis foi medida num
  stack quebrado. A inferência hoje está em 14 ms/imagem contra os 287 ms de
  antes; vale remedir `yolo26s-seg` antes de descartar.
- O colapso de recall atribuído ao excesso de background (15%) é confundido com
  a variação entre runs, que não havia sido medida. Hoje o background está em
  10% no treino.

---

## 7. Próximos passos

1. **Mais imagens anotadas** é o gargalo real: 124 no treino. As de maior
   retorno são cenas com pessoas pequenas e condições pouco representadas.
2. **`person` não sai por hiperparâmetro.** Com ruído de ±0,04 e cinco runs
   todas entre 0,11 e 0,27 de mAP50-95, o que resta é estrutural: mais dados, ou
   SAHI só na inferência (passe no frame inteiro para as cargas + passes em
   tiles para as pessoas, unidos por NMS). Diferente do Experimento 1 do
   relatório, que fatiava o dataset de treino e partia as cargas.
3. **`load` responde a resolução de treino.** O próximo passo é 1600, que exige
   o TTY.
4. **Remedir `yolo26s-seg`** no stack atual.
5. **Anotar mais `pipe` ou aceitar que ele não é medido.**
