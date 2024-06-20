import cv2
import numpy as np
import ImageProcessor

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

#se foloseste imread normal ca fac dupa schimbarea din BGR IN HSV
img = cv2.imread("./sample.jpeg")

masks = ImageProcessor.createColorMasks(img)
#salveaza doar pixelii dintr-o masca
result = cv2.bitwise_and(img, img, mask= masks[0])

# ACEL RESIZE WITH ASPECT RATIO SE FOLOSESTE DOAR PENTRU AFISARE
# IMAGINEA CARE SE PROCESEAZA TREBUIE SA RAMANA FARA RESIZE
result = ResizeWithAspectRatio(result, width=1280)
cv2.imshow('mask', result)

cv2.waitKey(0)
cv2.destroyAllWindows()