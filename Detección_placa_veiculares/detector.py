from ultralytics import YOLO
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\sgabr\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

class PlateDetector:
    def __init__(self, model_path="model/best.pt"):
        self.model = YOLO(model_path)

    def detect_plate(self, frame):
        results = self.model(frame, conf=0.4)[0]
        plates = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append((x1, y1, x2, y2))

        return plates

    def run_webcam(self, ocr=True):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ No se pudo abrir la webcam")
            return

        print("📷 Webcam activa — presiona Q para salir")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            plates = self.detect_plate(frame)

            for (x1, y1, x2, y2) in plates:
                plate_img = frame[y1:y2, x1:x2]

                text = ""
                if ocr:
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(
                        gray,
                        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    ).strip()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("ANPR - Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
