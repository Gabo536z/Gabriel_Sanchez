from detector import PlateDetector
from ocr import PlateOCR

# Ruta del modelo
MODEL_PATH = "model/best.pt"

# Inicializar detector y OCR
detector = PlateDetector(MODEL_PATH)
ocr = PlateOCR()

def main():
    print("📷 Iniciando detección de placas por webcam...")
    print("Presiona Q para salir")
    detector.run_webcam(ocr)

if __name__ == "__main__":
    main()
