import cv2
import numpy as np

#folosim valori HSV
redLowerBound = np.array([81, 53, 49])
redUpperBound = np.array([183, 84, 79])

greenLowerBound = np.array([15, 46, 30])
greenUpperBound = np.array([82, 255, 165])

blueLowerBound = np.array([73, 100, 139])
blueUpperBound = np.array([154, 191, 255])

params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 110
params.maxThreshold = 140
 
# Filter by Area.
params.filterByArea = True
params.minArea = 400
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.1
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.03

#se creeaza 3 sau mai multe masti pentru a obtine fiecare fuzibila
#in functie de culoarea sa si asa (poate reusesc sa setez unele rangeuri
#cat sa proceseze 2 culori but idk)

#apoi se combia toate acele masti intr-una
def createColorMasks(image):
    #imshow functioneaza pe BGR asa ca image nu mai poate fi afisat
    #si sa arate normal in continuare
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blured = cv2.blur(hsv_image,(15, 15))

    redMask = cv2.inRange(blured, redLowerBound, redUpperBound)
    greenMask = cv2.inRange(blured, greenLowerBound, greenUpperBound)
    blueMask = cv2.inRange(blured, blueLowerBound, blueUpperBound)

    detector = cv2.SimpleBlobDetector_create(params)
 
    # # Detect blobs.
    keypoints = detector.detect(image)

    # # Draw detected blobs as red circles for the redMask.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(redMask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('keypoi',  im_with_keypoints)

    # cv2.imshow('red', redMask)
    # cv2.imshow('green', greenMask)
    # cv2.imshow('blue', blueMask)

    return [redMask, greenMask, blueMask]