import cv2
import numpy as np
import os
import glob
import argparse

def process_sparse_optical_flow(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png")))
    
    if len(image_paths) < 2:
        return

    # Configuração dos algoritmos de detecção e rastreamento
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    color = np.random.randint(0, 255, (200, 3))

    old_frame = cv2.imread(image_paths[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    for i in range(1, len(image_paths)):
        frame = cv2.imread(image_paths[i])
        if frame is None:
            continue
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_out = frame.copy()

        if p0 is not None and len(p0) > 0:
            # Cálculo do Fluxo Óptico Esparso (Lucas-Kanade)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for j, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)
                    
                    frame_out = cv2.arrowedLine(frame_out, (c, d), (a, b), color[j % 200].tolist(), 2, tipLength=0.3)
                    frame_out = cv2.circle(frame_out, (a, b), 5, color[j % 200].tolist(), -1)

        base_name = os.path.basename(image_paths[i])
        out_path = os.path.join(output_dir, f"sparse_{base_name}")
        cv2.imwrite(out_path, frame_out, [cv2.IMWRITE_JPEG_QUALITY, 85])

        old_gray = frame_gray.copy()
        
        # Redetecção de pontos a cada frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula o Fluxo Óptico Esparso (Lucas-Kanade)")
    parser.add_argument("--input", default="operations/flow", help="Pasta com as imagens originais")
    parser.add_argument("--output", default="operations/flow_sparse_output", help="Pasta para salvar os resultados")
    args = parser.parse_args()
    process_sparse_optical_flow(args.input, args.output)
