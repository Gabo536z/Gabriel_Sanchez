import cv2
import os
from ultralytics import YOLO

# Cargar modelo YOLO
model = YOLO("yolov8n.pt")  # Puedes usar yolov8s.pt o tu modelo entrenado

# Activar cámara
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona la tecla 'c' para capturar la foto.")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error leyendo la cámara.")
        break

    cv2.imshow("CAMARA - Presiona 'c' para capturar", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Guardar frame temporal
        cv2.imwrite("captura_temp.jpg", frame)
        print("📸 Foto tomada. Analizando...")

        # Detectar objetos
        results = model("captura_temp.jpg")

        # Extraer la etiqueta del objeto más probable
        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes[0].cls[0])
            label = model.names[cls_id]
            print(f"Objeto detectado: {label}")

            # Crear carpeta si no existe
            folder = f"./{label}"
            os.makedirs(folder, exist_ok=True)

            # Guardar imagen final
            save_path = os.path.join(folder, "foto_detectada.jpg")
            cv2.imwrite(save_path, frame)

            print(f"📂 Imagen guardada en: {save_path}")

        else:
            print("No se detectó ningún objeto claro.")

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
