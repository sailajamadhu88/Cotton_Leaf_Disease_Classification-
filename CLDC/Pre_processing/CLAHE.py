import cv2

class CLAHE:
    def CE_CLAHE(self, spath, dpath):
        img = cv2.imread(spath)

        # Convert image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split LAB channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge channels back together
        lab_clahe = cv2.merge((l_clahe, a, b))

        # Convert LAB back to BGR color space
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        cv2.imwrite(dpath, img_clahe)
