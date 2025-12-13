import cv2
import pytesseract

class PlateOCR:

    def __init__(self):
        self.config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 40, 40)
        gray = cv2.equalizeHist(gray)
        return gray

    def read_plate(self, img):
        if img is None or img.size == 0:
            return ""

        proc = self.preprocess(img)
        text = pytesseract.image_to_string(proc, config=self.config)
        text = "".join([c for c in text if c.isalnum()]).upper()
        return text
