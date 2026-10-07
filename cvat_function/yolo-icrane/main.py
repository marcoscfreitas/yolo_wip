import base64
import io
import json
import os

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

PESOS = os.environ.get("ICRANE_PESOS", "/opt/nuclio/best.pt")

# `load` rende melhor a 1280; `person`/`pipe` a 1408 (docs/NOTAS_TECNICAS.md, secao 4).
PASSES = [dict(classes=[0], imgsz=1280), dict(classes=[1, 2], imgsz=1408)]


def init_context(context):
    context.user_data.model = YOLO(PESOS)
    context.user_data.device = os.environ.get("ICRANE_DEVICE", 0)


def handler(context, event):
    data = event.body
    img = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    thr = float(data.get("threshold", 0.5))
    model = context.user_data.model

    out = []
    for p in PASSES:
        r = model.predict(img, conf=thr, device=context.user_data.device,
                          verbose=False, **p)[0]
        if r.masks is None:
            continue
        for poly, conf, k in zip(r.masks.xy, r.boxes.conf.tolist(),
                                 r.boxes.cls.int().tolist()):
            if len(poly) < 3:
                continue
            # contornos crus tem centenas de pontos; simplifica para facilitar a correcao
            poly = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), 1.5, True).reshape(-1, 2)
            if len(poly) < 3:
                continue
            out.append({"confidence": str(conf), "label": model.names[k],
                        "points": poly.flatten().tolist(), "type": "polygon"})

    return context.Response(body=json.dumps(out), headers={},
                            content_type="application/json", status_code=200)
