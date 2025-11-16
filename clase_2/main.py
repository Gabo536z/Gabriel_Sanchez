import cv2
import os
from ultralytics import YOLO

# Cargar modelo YOLO
model = YOLO("yolov8n.pt")

# Traducción de nombres al español
TRADUCCION = {
    "cell phone": "celular",
    "laptop": "portatil",
    "keyboard": "teclado",
    "mouse": "mouse",
    "tv": "televisor",
    "car": "carro",
    "motorcycle": "moto",
    "bottle": "botella",
    "cup": "taza",
    "backpack": "maleta",
    "book": "libro",
    "chair": "silla",
    "person": "persona"
}

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
        cv2.imwrite("captura_temp.jpg", frame)
        print("📸 Foto tomada. Analizando...")

        # Detectar objetos
        results = model("captura_temp.jpg")

        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes[0].cls[0])
            label_ing = model.names[cls_id]   # inglés

            # Si detecta persona → NO guardar
            if label_ing == "person":
                print("❌ No se pueden tomar fotos de personas.")
                continue

            # Traducir al español
            label_esp = TRADUCCION.get(label_ing, label_ing)

            # Crear carpeta
            folder = f"./{label_esp}"
            os.makedirs(folder, exist_ok=True)

            # Contar fotos existentes (máximo 5)
            fotos_actuales = len(os.listdir(folder))
            if fotos_actuales >= 5:
                print(f"⚠ Ya existen 5 fotos de '{label_esp}'. No se guardará más.")
                continue

            # Guardar imagen
            save_path = os.path.join(folder, f"{label_esp}_{fotos_actuales+1}.jpg")
            cv2.imwrite(save_path, frame)

            print("📂 Imagen guardada correctamente.")

        else:
            print("No se detectó ningún objeto claro.")

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
