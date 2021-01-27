from scipy.spatial import distance as dist
# SciPy est un projet visant à unifier et fédérer un ensemble de bibliothèques Python à usage scientifique.
# Scipy utilise les tableaux et matrices du module NumPy donc il depend de NumPy
# Scipy.spatial peut calculer des triangulations, des diagrammes de Voronoi et
# des coques convexes d'un ensemble de points, en utilisant la bibliothèque Qhull .
# De plus, il contient des KDTree implémentations pour les requêtes de points du voisin le plus proche
# et des utilitaires pour les calculs de distance dans diverses métriques.
# distance permet le calcul de la matrice de distance à partir d'une collection de vecteurs d'observation
# bruts stockés dans un tableau rectangulaire.
from collections import OrderedDict
# OrderedDict sont des dictionnaires qui possèdent des capacités supplémentaires pour s'ordonner
#ils ont ont été conçus pour être performants dans les opérations de ré-arrangement.
# L'occupation mémoire, la vitesse de parcours et les performances de mise à jour étaient secondaires
import numpy as np
# NumPy est une extension du langage de programmation Python,
# destinée à manipuler des matrices ou tableaux multidimensionnels ainsi que
# des fonctions mathématiques opérant sur ces tableaux

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialise l'ID d'objet unique suivant avec deux
        # dictionnaires utilisés pour suivre le mappage d'un objet donné
        # ID à son point central et nombre d'images consécutives qu'il a
        # a été marqué comme "disparu", respectivement
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # stocke le nombre d'images consécutives maximum pour une donnée
        # objet peut être marqué comme "disparu" jusqu'à ce que nous
        # aurons plus besoin de l'enregistrer dans la suite du traitement
        self.maxDisappeared = maxDisappeared

        # stocker la distance maximale entre les points centrals  à associer
        # un objet - si la distance est supérieure à ce maximum
        # distance, nous commencerons à marquer l'objet comme "disparu"
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        # lors de l'enregistrement d'un objet, nous utilisons le prochain objet disponible
        # ID pour stocker le point central
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  # CHANGE
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # pour désenregistrer un ID d'objet, nous supprimons l'ID d'objet de
        # nos deux dictionnaires respectifs
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  # CHANGE

    def update(self, rects):
        # vérifier pour voir si la liste des rectangles de boîte englobante d'entrée
        # est vide
        if len(rects) == 0:
            # boucle sur tous les objets suivis existants et marque-les
            # comme disparu
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # si nous avons atteint un nombre maximum de
                # cadres où un objet donné a été marqué comme
                # manquant, désenregistrer(deregister)
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # revenir tôt car il n'y a pas de point central ou d'informations de suivi
            # mettre à jour
            # return self.objects
            return self.bbox

        # initialise un tableau de centroïdes(points centrals) d'entrée pour l'image courante
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
        # boucle sur les rectangles de la boîte englobante
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # utiliser les coordonnées de la boîte englobante pour dériver le point central
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])  # CHANGE

        # si nous ne suivons actuellement aucun objet, prenez l'entrée
        # centroids(point centraux) et enregistrez chacun d'eux
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE

        # sinon, des objets sont qui sont en suivi, alors nous devons donc
        # essayez de faire correspondre les point centraux d'entrée à l'objet existant
        # centroids
        else:
            # récupérer l'ensemble des ID d'objet et des centres de gravité correspondants
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # calculer la distance entre chaque paire d'objets
            # centroids et centroids d'entrée, respectivement - notre
            # objectif sera de faire correspondre un point central d'entrée à un
            # centre de l'objet
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # pour effectuer cette correspondance, nous devons (1) trouver la
            # plus petite valeur de chaque ligne, puis (2) trier la ligne
            # index en fonction de leurs valeurs minimales afin que la ligne
            # avec la plus petite valeur seront stockées en tête de la
            # liste
            rows = D.min(axis=1).argsort()

            # ensuite, nous effectuons un processus similaire sur les colonnes en
            # trouver la plus petite valeur dans chaque colonne puis
            # trier à l'aide de la liste d'index des lignes précédemment calculée
            cols = D.argmin(axis=1)[rows]

            # afin de déterminer si nous devons mettre à jour, enregistrer(register function),
            # ou désenregistrer(deregister function) un objet dont nous devons garder la trace
            # des lignes et des index de colonnes que nous avons déjà examinés
            usedRows = set()
            usedCols = set()

            # boucle sur la combinaison de l'index (ligne, colonne)
            # tuples
            for (row, col) in zip(rows, cols):
                # si nous avons déjà examiné la ligne ou
                # valeur de colonne avant, on l'ignore
                if row in usedRows or col in usedCols:
                    continue

                # si la distance entre les points centraux est supérieure à
                # la distance maximale, n'associez pas les deux
                # centroids sur le même objet
                if D[row, col] > self.maxDistance:
                    continue

                # sinon, récupérez l'ID objet de la ligne courante,
                # définit son nouveau point central et réinitialise le
                # compteur
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0

                # indique que nous avons examiné chacune des lignes et
                # index de colonnes, respectivement
                usedRows.add(row)
                usedCols.add(col)

            # calculer à la fois l'index de ligne et de colonne que nous n'avons PAS encore
            # examiné
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # dans le cas où le nombre de points central d'objet est
            # égal ou supérieur au nombre du point central d'entrée
            # nous devons vérifier et voir si certains de ces objets ont
            # potentiellement disparu
            if D.shape[0] >= D.shape[1]:
                # boucle sur les index de ligne inutilisés
                for row in unusedRows:
                    # récupérer l'ID d'objet de la ligne correspondante
                    # indexer et incrémenter le compteur disparu
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # vérifier si le nombre de
                    # frames l'objet a été marqué "disparu"
                    # pour les mandats de désenregistrement de l'objet
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # sinon, si le nombre des point centraux d'entrée est plus grand
            # que le nombre des points centraux d'objets existants dont nous avons besoin
            # enregistrer(register function) chaque nouveau point central d'entrée comme un objet traçable
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        # renvoie l'ensemble des objets traçables
        # return self.objects
        return self.bbox

