import cv2
import time
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0) #affichage video du webcam pc

start_time = time.time() #obtenir le temps d'exécution du programme

# initiation pour calcul fps
display_time = 2
fc = 0
FPS = 0

#creation trackbar
cv2.namedWindow("Trackbars") #création d'une fenêtre qui va être utilisée comme emplacement pour les trackbars
cv2.createTrackbar("Low-H", "Trackbars", 94, 180, nothing) #creation trackbar. Parametres(nom_parametre, nom fenetre,
# position curseur, valeur max, fonction à appeler à chaque fois que le curseur change de position)
cv2.createTrackbar("Low-S", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("Low-V", "Trackbars", 2, 255, nothing)
cv2.createTrackbar("High-H", "Trackbars", 126, 180, nothing)
cv2.createTrackbar("High-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("High-V", "Trackbars", 255, 255, nothing)

font = cv2.FONT_HERSHEY_SIMPLEX #initiation police

while True:

    _, frame = cap.read() #charger l'image de la webcam
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #conversion vers espace hsv
    frame_hsv_blur = cv2.GaussianBlur(hsv_frame, (7, 7), 0)
    kernel = np.ones((5, 5), np.float32) / 25
    filtered_frame = cv2.filter2D(frame_hsv_blur, -1, kernel)



    # creation plage de la couleur bleue
    low_blue = np.array([94, 80, 2]) #val min
    high_blue = np.array([120, 255, 255]) #val max
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue) #creation mask pour seuillage. Parametres:(source,val min, val max)

    # creation plage de la couleur verte
    low_green = np.array([25,52,72])
    high_green = np.array([102,255,255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)

    #configuration des trackbars
    Low_H =cv2.getTrackbarPos("Low-H", "Trackbars")
    Low_S = cv2.getTrackbarPos("Low-S", "Trackbars")
    Low_V = cv2.getTrackbarPos("Low-V", "Trackbars")
    High_H = cv2.getTrackbarPos("High-H", "Trackbars")
    High_S = cv2.getTrackbarPos("High-S", "Trackbars")
    High_V = cv2.getTrackbarPos("High-V", "Trackbars")

    # définition de la plage des couleur pour le masque
    low = np.array([Low_H, Low_S, Low_V])
    high = np.array([High_H, High_S, High_V])
    mask = cv2.inRange(hsv_frame, low, high)

    #supression bruit par morphologie mathematique
    kernel = np.ones( (5,5) , np.uint8) #definition noyau ou element structural.
    # Parametres (forme,data type= unsigned integer 8bits)

    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)


    #detection contours de la couleur bleue
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#retourne les contours et un vecteur
    #contenant des informations sur la topologie de l'image. Parametres: (image, mode = RETR_TREE récupère tous les contours
    # et crée une liste complète de la hiérarchie,Méthode d'approximation des contours=CHAIN_APPROX_SIMPLE pour gagner de memoire)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour) #calcul surface
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)  # approximation des contours pour avoir des lignes droites
        # obtention positions du texte dans l'image
        x = approx.ravel()[0]  # retourne tableau applati
        y = approx.ravel()[1]
        if (area > 400): #filtrage des contours petits/bruits
            cv2.drawContours(frame, [approx], 0, (0, 0, 0),
                             5)  # Parametres:(image,contour, contourIdx: indique quel contour à dessiner,
            # couleur contour,épaisseur)
            if len(approx) == 3:  # si on a 3 sommets => Triangle
                cv2.putText(frame, "Triangle bleu ", (x, y), font, 1, (0, 0, 0))  # texte dans l'image
            elif len(approx) == 4:
                cv2.putText(frame, "Rectangle bleu ", (x, y), font, 1, (0, 0, 0))
            elif 10 < len(approx) < 20:
                cv2.putText(frame, "Disque bleu", (x, y), font, 1, (0, 0, 0))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # de même pour la couleur verte
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)  # calcul surface
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),
                                  True)  # approximation des contours pour avoir des lignes droites
        # obtention positions du texte dans l'image
        x = approx.ravel()[0]  # retourne tableau applati
        y = approx.ravel()[1]
        if (area > 400):  # filtrage des contours petits/bruits
            cv2.drawContours(frame, [approx], 0, (0, 0, 0),
                             5)  # Parametres:(image,contour, contourIdx: indique quel contour à dessiner,
            # couleur contour,épaisseur)
            if len(approx) == 3:  # si on a 3 sommets => Triangle
                cv2.putText(frame, "Triangle vert ", (x, y), font, 1, (0, 0, 0))  # texte dans l'image
            elif len(approx) == 4:
                cv2.putText(frame, "Rectangle vert ", (x, y), font, 1, (0, 0, 0))
            elif 10 < len(approx) < 20:
                cv2.putText(frame, "Disque vert", (x, y), font, 1, (0, 0, 0))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    #calcul fps
    fc += 1
    TIME = time.time() - start_time

    if (TIME) >= display_time:
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()

    # afficher le nombre de fps
    fps_disp = "FPS: " + str(FPS)[:5]
    image = cv2.putText(frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #affichage
    cv2.imshow("frame", frame)
    cv2.imshow("video segmentee", mask)
    key = cv2.waitKey(1) & 0xFF

    # appuyez sur esc pour quitter le streaming
    if key == 27:
        break

cv2.destroyAllWindows()