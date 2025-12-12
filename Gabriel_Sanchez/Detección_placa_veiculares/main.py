from detector import PlateDetector
from ocr import PlateOCR
import cv2
import os

# Rutas
MODEL_PATH = "model/best.pt"
IMAGE_FOLDER = "images/"
OUTPUT_FOLDER = "results/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

detector = PlateDetector(MODEL_PATH)
ocr = PlateOCR()

def process_folder():
    print("Procesando imágenes de la carpeta...")
    for imgname in os.listdir(IMAGE_FOLDER):
        path = os.path.join(IMAGE_FOLDER, imgname)

        img = cv2.imread(path)
        if img is None:
            print("No se pudo abrir:", imgname)
            continue

        plates = detector.detect_plates(img)

        for (x1, y1, x2, y2) in plates:
            plate_crop = img[y1:y2, x1:x2]
            text = ocr.read_plate(plate_crop)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, text, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            print(f"{imgname} → {text}")

        cv2.imwrite(f"{OUTPUT_FOLDER}/OUT_{imgname}", img)

def main():
    print("1) Procesar carpeta\n2) Webcam")
    op = input("Elige opción: ")

    if op == "1":
        process_folder()
    else:
        detector.run_webcam(ocr)

if __name__ == "__main__":
    main()
