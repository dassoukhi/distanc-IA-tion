#Avant d'exécuter ce fichier, assurez-vous que vous avez le dossier Pretrained model et classes
#vous pouvez tout obtenir via mon lien github que je mentionnerai ci-dessous
# Cela dépendra des performances du processeur si les performances de votre processeur sont bonnes,
# la vidéo sera traitée rapidement # je ne l'ai pas bonne performance cpu à ce pourquoi son traitement assez faible
import cv2
# openCV est utilisé pour toutes sortes d'analyses d'images et de vidéos,
# telles que la reconnaissance et la détection faciales, la lecture de plaques d'immatriculation,
# l'édition de photos, la vision robotique avancée, la reconnaissance optique de caractères, et bien plus encore.
import datetime
# Le module datetime permet de manipuler les dates et les heures.
import imutils
# imutis compose une série de fonctions pratiques pour rendre les fonctions de traitement d'image de base telles que
# la traduction,la rotation, le redimensionnement, le squelette, l'affichage des images Matplotlib, le tri des contours,
# la détection des bords et bien plus encore avec OpenCV et Python .

import numpy as np
# NumPy est une extension du langage de programmation Python,
# destinée à manipuler des matrices ou tableaux multidimensionnels ainsi que
# des fonctions mathématiques opérant sur ces tableaux
from centroidtracker import CentroidTracker
from itertools import combinations
#Ce module implémente de nombreuses briques d'itérateurs rapides et efficaces pour boucler efficacement
import math
# Ce module math permet d'accéder aux fonctions mathématiques définies par le standard C


#Le modèle et les fichiers prototypes sont ici
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"

# modelConfiguration = "C:\\Users\\Dass\Desktop\\test\\Computer-Vision\\Social_distancing\\yolov3.cfg"
# modelWeight = "C:\\Users\\Dass\\Desktop\\test\\Computer-Vision\\Social_distancing\\yolov3.weights"
#
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# detector = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

#mention du nombre de classes ici
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))
#Passez le lien vidéo ici

def main():
    #ouverture de la video avec openCv

    cap = cv2.VideoCapture('vid_short.mp4')

    #cap = cv2.VideoCapture(0)

    #prendre l'heure actuelle d'images par seconde avant l'execution
    fps_start_time = datetime.datetime.now()
    # fps = images par seconde // initié à 0
    fps = 0
    # nombre total d'images
    total_frames = 0

# boucle de traitement de la video
    while True:
        # ici on la video image par image
        ret, frame = cap.read()
        # on redimensionne l'image d'une largeur de 600 px
        frame = imutils.resize(frame, width=600)
        # on increment total_frames de 1
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        centroid_dict = dict()
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)


            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

            # text = "ID: {}".format(objectId)
            # cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for id, box in centroid_dict.items():
            if id in red_zone_list:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)


        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Social_Distancing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

main()