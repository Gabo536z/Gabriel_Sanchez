import cv2
import os
import time
from datetime import datetime
import platform


if platform.system() == "Windows":
    try:
        import winsound
        def play_click():
            winsound.Beep(750, 100)
    except Exception:
        def play_click():
            print("\a", end='', flush=True)
else:
    def play_click():
        print("\a", end='', flush=True)

from detector_formas import detectar_formas

# Carpeta base
BASE_SAVE_DIR = "formas"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# CONTAR IMÁGEN
def contar_imagenes(forma):
    carpeta = os.path.join(BASE_SAVE_DIR, forma)
    if not os.path.exists(carpeta):
        return 0
    return len([n for n in os.listdir(carpeta) if n.lower().endswith((".png", ".jpg", ".jpeg"))])

# GUARDAR IMAGEN 
def guardar_imagen(frame, forma, bbox):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta = os.path.join(BASE_SAVE_DIR, forma)
    os.makedirs(carpeta, exist_ok=True)

    if bbox is None:
        filename = f"{forma}_{now}.jpg"
        path = os.path.join(carpeta, filename)
        cv2.imwrite(path, frame)
        return path

    x, y, w, h = bbox
    pad = int(0.08 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    crop = frame[y1:y2, x1:x2]
    filename = f"{forma}_{now}.jpg"
    path = os.path.join(carpeta, filename)
    cv2.imwrite(path, crop)
    return path

# FUNCIÓN SELECCIÓN DE FORMA
def seleccionar_forma():
    print("\n=== Selecciona la forma a capturar ===")
    print("[1] Triángulo")
    print("[2] Cuadrado")
    print("[3] Círculo")

    while True:
        opcion = input("Elige una opción (1-3): ")

        if opcion == "1":
            return "triangulo"
        elif opcion == "2":
            return "cuadrado"
        elif opcion == "3":
            return "circulo"
        else:
            print("❌ Opción inválida. Intenta nuevamente.")

# PROGRAMA PRINCIPAL
def main():
    forma_objetivo = seleccionar_forma()
    print(f"\n✔ Forma seleccionada: {forma_objetivo}\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo la cámara.")
            break

        frame_detectado, forma_detectada, bbox = detectar_formas(frame)

        # Mostrar en pantalla la forma seleccionada
        cv2.putText(frame_detectado, f"Buscando: {forma_objetivo}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame_detectado, f"Detectada: {forma_detectada}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.putText(frame_detectado,
                    "[C] Capturar  |  [F1-F3] Cambiar forma  |  [Q] Salir",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 2)

        cv2.imshow("Detector de Formas", frame_detectado)

        key = cv2.waitKey(1) & 0xFF

        # TECLAS PARA CAMBIAR FORMA
        if key == 0:  # F1 (triángulo)
            forma_objetivo = "triangulo"
            print("✔ Forma cambiada a: TRIÁNGULO")
            play_click()

        elif key == 1:  # F2 (cuadrado)
            forma_objetivo = "cuadrado"
            print("✔ Forma cambiada a: CUADRADO")
            play_click()

        elif key == 2:  # F3 (círculo)
            forma_objetivo = "circulo"
            print("✔ Forma cambiada a: CÍRCULO")
            play_click()

        # TOMAR CAPTURA
        if key == ord('c'):
            if forma_detectada != forma_objetivo:
                print(f"❌ No se puede capturar. Detectado: {forma_detectada} — Se espera: {forma_objetivo}")
                play_click()
                continue

            path = guardar_imagen(frame, forma_detectada, bbox)
            print(f"✔ Imagen guardada en: {path}")
            play_click()

        # SALIR DEL PROGRAMA
        if key == ord('q') or key == 27:
            print("Saliendo...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
