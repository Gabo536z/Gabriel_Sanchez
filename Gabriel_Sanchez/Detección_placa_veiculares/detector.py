from ultralytics import YOLO
import cv2

class PlateDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_plates(self, frame):
        results = self.model.predict(frame, conf=0.35, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append((x1,y1,x2,y2))
        return detections

    def run_webcam(self, ocr):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo abrir la cámara.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            plates = self.detect_plates(frame)
            for (x1,y1,x2,y2) in plates:
                plate_crop = frame[y1:y2, x1:x2]
                text = ocr.read_plate(plate_crop)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, text, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("ANPR - Webcam", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
