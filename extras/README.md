# Extras

Experimentos que **não fazem parte do pipeline YOLO**. Não importam
`ultralytics` nem tocam o `yolo_dataset`: recebem uma pasta de imagens e
escrevem visualizações de fluxo óptico.

- `optical_flow.py` — fluxo óptico denso (Farnebäck), saída em HSV
- `optical_flow_sparse.py` — fluxo óptico esparso (Lucas-Kanade)

Guardados porque movimento é um gerador de candidatos barato para `person`, que
é a classe difícil deste projeto (mediana de 20 px na validação, e o mAP não sai
do lugar por hiperparâmetro). Se essa linha for descartada, a pasta pode ir
embora inteira.
