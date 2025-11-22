import cv2
import numpy as np
import math

def detectar_formas(frame, min_area=400):
    """
    Detecta triángulo, cuadrado y círculo.
    Devuelve:
        - frame anotado
        - nombre de la forma detectada
        - bounding box (x, y, w, h)
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 1)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 10
    )

    # Buscar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        return frame, "desconocida", None

    # Elegir el contorno más grande
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    forma_detectada = "desconocida"
    bbox = None

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)

        # Calcular circularidad
        circularidad = 0.0
        if peri > 0:
            circularidad = (4 * math.pi * area) / (peri * peri)

        vertices = len(approx)

        # Clasificación
        if vertices == 3:
            forma = "triangulo"

        elif vertices == 4:
            ar = w / float(h)
            if 0.80 <= ar <= 1.25:
                forma = "cuadrado"
            else:
                forma = "cuadrado" 

        else:
            # filtrado de círculos más robusto
            forma = "circulo" if circularidad > 0.70 else "desconocida"

        if forma != "desconocida":
            forma_detectada = forma
            bbox = (x, y, w, h)

            # Dibujar contornos y anotaciones
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, f"{forma_detectada} ({circularidad:.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            break

    return frame, forma_detectada, bbox
