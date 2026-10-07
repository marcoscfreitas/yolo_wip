# Pré-anotação no CVAT com o modelo da lança

Função Nuclio que publica o modelo treinado (`load`, `person`, `pipe`) como
detector de segmentação no CVAT. Em **Actions → Automatic annotation** a imagem
inteira é anotada de uma vez e só resta corrigir.

Vale para a **câmera da ponta da lança**. Não serve para a câmera do laydown.

## Arquivos

| arquivo | função |
|---|---|
| `yolo-icrane/function-gpu.yaml` | definição da função: imagem, dependências, GPU, classes |
| `yolo-icrane/main.py` | `init_context` carrega o modelo; `handler` devolve polígonos |
| `yolo-icrane/best.pt` | pesos — **não versionado** (`*.pt` está no `.gitignore`) |

Para montar a pasta, copie para `yolo-icrane/` o `best.pt` desejado, por exemplo
`runs/segment/treinamentos/res_1408/weights/best.pt`.

O `main.py` roda cada classe na resolução em que ela rende melhor (`load` a 1280,
`person`/`pipe` a 1408; ver `docs/NOTAS_TECNICAS.md`, seção 4) e simplifica os
contornos com `approxPolyDP` para facilitar a correção.

## Implantação no servidor do CVAT

Pré-requisitos: CVAT com o Nuclio no ar (compose com
`components/serverless/docker-compose.serverless.yml`), NVIDIA Container Toolkit e
CDI configurados.

No PC, copie a pasta (o `scp` roda **no PC**, onde os arquivos estão):

```bash
ssh USUARIO@SERVIDOR "mkdir -p ~/cvat/serverless/pytorch/icrane/yolo/nuclio"
scp cvat_function/yolo-icrane/* USUARIO@SERVIDOR:~/cvat/serverless/pytorch/icrane/yolo/nuclio/
```

No servidor:

```bash
cd ~/cvat
./serverless/deploy_gpu.sh serverless/pytorch/icrane/yolo/nuclio
nuctl get functions --platform local     # pth-icrane-yolo-seg deve estar `ready`
docker exec $(docker ps -q -f name=pth-icrane-yolo-seg) \
  python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A saída do último comando deve ser `2.5.1+cu121 True`.

## Armadilhas já encontradas

- **`libGL.so.1` ausente.** O `ultralytics` instala o `opencv-python` completo e
  sobrescreve o `headless`. Por isso a imagem instala `libgl1` e `libglib2.0-0`.
- **`torch.cuda.is_available()` falso.** Sem fixar o torch, o `pip` traz a versão
  mais nova (CUDA 13), e o driver do servidor suporta só CUDA 12.2. O yaml instala
  `torch==2.5.1` e `torchvision==0.20.1` do índice `cu121` **antes** do
  ultralytics. Se o driver for atualizado, esse pino pode ser revisto.
- **`ultralytics==8.4.121`** é a versão do treino; manter igual na função.
- **`nuctl` 1.13.0 contra dashboard 1.16.3.** Versões diferentes; funciona, mas
  vale alinhar se aparecer comportamento estranho no deploy.

## Atualizar o modelo depois de um retreino

```bash
scp runs/segment/treinamentos/NOVO/weights/best.pt \
  USUARIO@SERVIDOR:~/cvat/serverless/pytorch/icrane/yolo/nuclio/best.pt
# no servidor:
cd ~/cvat && ./serverless/deploy_gpu.sh serverless/pytorch/icrane/yolo/nuclio
```

O build reaproveita as camadas do `pip` em cache; só os pesos mudam. Guarde o
`best.pt` anterior fora dessa pasta, que entra inteira na imagem. Se as classes
mudarem, atualize o `spec` no `function-gpu.yaml`; os labels da task no CVAT
precisam ter exatamente esses nomes.

## Uso no CVAT

**Actions → Automatic annotation**, escolher *iCrane YOLO seg (lanca)*, mapear
`load → load` e `person → person` (`pipe` é opcional). Confiança sugerida: **0,25
a 0,3**. Na validação, o melhor F1 fica em 0,32 para `load` e 0,25 para `person`;
a 0,5 o `person` acha só 19% das pessoas. Apagar um falso positivo custa menos que
desenhar uma pessoa que faltou.

Sem campo de limiar no diálogo, a função usa 0,5 (padrão do `main.py`).

## Persistência após reboot

O `cvat.service` precisa subir os dois arquivos compose (base + serverless) no
`ExecStart` e no `ExecStop`, e o `docker` precisa estar habilitado no boot. As
funções voltam sozinhas pelo `restartPolicy: always`.

## Disco

Cada função de GPU ocupa ~14 GB (PyTorch com CUDA), e o cache de build duplica
isso. `docker builder prune -a` libera o cache sem afetar o que está rodando.
Não usar `docker volume prune` nem `docker image prune -a`: o primeiro pode
apagar dados do CVAT e o segundo as imagens base do Nuclio.
