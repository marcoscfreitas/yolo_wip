import cv2
import numpy as np
import os
import glob
import argparse

def process_optical_flow(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png")))
    
    if len(image_paths) < 2:
        return

    # Preparação do primeiro frame
    frame1 = cv2.imread(image_paths[0])
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    for i in range(1, len(image_paths)):
        frame2 = cv2.imread(image_paths[i])
        if frame2 is None:
            continue
            
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Cálculo do Fluxo Óptico Denso (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None,
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        combined = np.hstack((frame2, bgr_flow))

        base_name = os.path.basename(image_paths[i])
        out_path = os.path.join(output_dir, f"flow_result_{base_name}")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 85])

        prvs = next_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula o Fluxo Óptico Denso (Farneback)")
    parser.add_argument("--input", default="operations/flow", help="Pasta com as imagens originais")
    parser.add_argument("--output", default="operations/flow_output", help="Pasta para salvar os resultados")
    args = parser.parse_args()
    process_optical_flow(args.input, args.output)
