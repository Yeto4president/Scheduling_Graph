import numpy as np
from prettytable import PrettyTable
import copy
import pyfiglet
import os

def log(message=""):
    global out_file
    out_file.write(message + "\n")


class Graphe:
    def __init__(self):
        self.tableauContrainte = None
        self.matriceAdj = []
        self.matriceVal = []
        self.entree = -1
        self.sortie = -1
        self.matriceCopiee = []
        self.tableauRangs = []
        self.listDateDouble = [[], [], [], [], []]
        self.listComplete = []
        self.listCheminCritique = []

    def readFile(self, chemin):
        tableauContrainte = []
        with open(chemin, "r") as fichier:
            for ligne in fichier:
                elements = ligne.split()
                if len(elements) < 2:
                    continue
                tache = int(elements[0])
                duree = int(elements[1])
                contraintes = [int(x) for x in elements[2:]] if len(elements) > 2 else []
                tableauContrainte.append([tache, duree, contraintes])
        self.tableauContrainte = tableauContrainte

    def affichageGrapheTriplets(self):
        n_sommets = len(self.matriceAdj)
        arcs = []
        for i in range(n_sommets):
            for j in range(n_sommets):
                if self.matriceAdj[i][j] == 1:
                    valeur = self.matriceVal[i][j]
                    try:
                        valeur = int(valeur)
                    except (ValueError, TypeError):
                        valeur = 0
                    arcs.append((i, j, valeur))
        n_arcs = len(arcs)
        log(" Création du graphe d’ordonnancement :")
        log(f"{n_sommets} sommets")
        log(f"{n_arcs} arcs")
        for (src, dest, poids) in arcs:
            log(f"{src} -> {dest} = {poids}")
        log()

    def matriceValeur(self):
        self.tableauContrainte = np.array(self.tableauContrainte, dtype=object)
        self.matriceVal = []
        n = len(self.tableauContrainte)
        for i in range(n + 1):
            ligne = ['-']
            terminaux = True
            for j in range(n):
                if i == 0:
                    terminaux = False
                    if len(self.tableauContrainte[j][2]) == 0:
                        ligne.append(0)
                    else:
                        ligne.append('-')
                elif self.tableauContrainte[j][2].__contains__(i):
                    ligne.append(self.tableauContrainte[i - 1][1])
                    terminaux = False
                else:
                    ligne.append('-')
            ligne.append('-' if not terminaux else self.tableauContrainte[i - 1][1])
            self.matriceVal.append(ligne)
        self.matriceVal.append(['-' for _ in range(n + 2)])
        tab_val = PrettyTable()
        tab_field_names = ["\\", "α"] + [str(i) for i in range(1, len(self.matriceVal) - 1)] + ["ω"]
        tab_val.field_names = tab_field_names
        tab_tmp = np.array(["α"] + [str(i) for i in range(1, len(self.matriceVal) - 1)] + ["ω"])
        matriceVal = np.column_stack((tab_tmp, self.matriceVal))
        for line in matriceVal:
            tab_val.add_row(line)
        log(tab_val.get_string())
        log()

    def matriceAdjacence(self):
        self.tableauContrainte = np.array(self.tableauContrainte, dtype=object)
        n = len(self.tableauContrainte)
        self.matriceAdj = []
        for i in range(n + 1):
            ligne = [0]
            terminaux = True
            for j in range(n):
                if i == 0:
                    terminaux = False
                    ligne.append(1 if not self.tableauContrainte[j][2] else 0)
                elif i in self.tableauContrainte[j][2]:
                    ligne.append(1)
                    terminaux = False
                else:
                    ligne.append(0)
            ligne.append(0 if not terminaux else 1)
            self.matriceAdj.append(ligne)
        self.matriceAdj.append([0] * (n + 2))
        self.matriceAdj = np.array(self.matriceAdj, dtype=int)
        tab_adj = PrettyTable()
        tab_field_names = ["\\", "α"] + [str(i) for i in range(1, len(self.matriceAdj) - 1)] + ["ω"]
        tab_adj.field_names = tab_field_names
        tab_tmp = np.array(["α"] + [str(i) for i in range(1, len(self.matriceAdj) - 1)] + ["ω"])
        matriceAdj = np.column_stack((tab_tmp, self.matriceAdj))

    def detectionCircuitEtPoints(self):
        def PointEntree():
            compteur = 0
            point = None
            n = len(self.matriceAdj)
            for i in range(n):
                if not any(self.matriceAdj[j][i] == 1 for j in range(n)):
                    compteur += 1
                    point = i
            return compteur, point

        def PointSortie():
            compteur = 0
            point = None
            n = len(self.matriceAdj)
            for i in range(n):
                if not any(self.matriceAdj[i][j] == 1 for j in range(n)):
                    compteur += 1
                    point = i
            return compteur, point

        nb_entree, point_entree = PointEntree()
        nb_sortie, point_sortie = PointSortie()
        self.entree = point_entree
        self.sortie = point_sortie
        log(">> Points d'entrée et de sortie :")
        log(f"Il y a un seul point d’entrée : {self.entree}")
        log(f"Il y a un seul point de sortie : {self.sortie}")
        log()

        log(">> Détection de circuit (méthode d’élimination des points d’entrée) :")
        n = len(self.matriceAdj)
        copie = np.copy(self.matriceAdj)
        total_sommets = set(range(n))
        iteration = 1
        while True:
            points_entree = [i for i in range(n) if i in total_sommets and not any(copie[j][i] == 1 for j in range(n))]
            if points_entree:
                log(f"Points d’entrée (Itération {iteration}) : {' '.join(str(p) for p in points_entree)}")
                log("Suppression des points d’entrée.")
                for p in points_entree:
                    copie[p, :] = 2
                    copie[:, p] = 2
                    total_sommets.discard(p)
                if total_sommets:
                    log("Sommets restants : " + " ".join(str(s) for s in sorted(total_sommets)))
                else:
                    log("Sommets restants : Aucun")
                    break
            else:
                log("Aucun point d'entrée trouvé alors que des sommets restent.")
                break
            iteration += 1
        if total_sommets:
            log("-> ERREUR : Circuit détecté. Ordonnancement impossible !")
            log()
            return -1
        else:
            log("-> Aucun circuit détecté.")
            log()
        return 0

    def calculRang(self):
        log(">> Calcul des rangs des sommets :")
        matriceCopie = np.copy(self.matriceAdj)
        etatRestant = len(self.matriceAdj)
        rang = 0
        while etatRestant > 0:
            eliminer = []
            log(f"Recherche des sommets de rang {rang} ...")
            for i in range(len(matriceCopie)):
                contientUn = False
                nbDeux = 0
                for j in range(len(matriceCopie)):
                    if matriceCopie[j][i] == 2:
                        nbDeux += 1
                    if matriceCopie[j][i] == 1:
                        contientUn = True
                        break
                if not contientUn and nbDeux != len(matriceCopie):
                    eliminer.append(i)
                    etatRestant -= 1
            self.tableauRangs.append(list(eliminer))
            log(f"Sommets éliminés au rang {rang} : " + "{ " + " ".join("α" if s == 0 else ("ω" if s == len(self.matriceAdj) - 1 else str(s)) for s in eliminer) + " }")
            for s in eliminer:
                for n in range(len(matriceCopie)):
                    matriceCopie[s][n] = 2
                for n in range(len(matriceCopie)):
                    matriceCopie[n][s] = 2
            rang += 1
        log(">> Calcul des rangs terminé.")
        log()

    def auPlusTot(self):
        log(">> Calcul du calendrier au plus tôt :")
        for i in range(len(self.tableauRangs)):
            for j in range(len(self.tableauRangs[i])):
                self.listDateDouble[0].append(i)
        for i in range(len(self.tableauRangs)):
            taches = []
            for s in self.tableauRangs[i]:
                taches.append(s)
                if s == 0 or s == len(self.matriceAdj) - 1:
                    taches.append(0)
                    log(f"Sommet {s} (entrée/sortie) : durée fixée à 0")
                else:
                    for ligne in self.tableauContrainte:
                        if ligne[0] == s:
                            taches.append(ligne[1])
                            log(f"Sommet {s} : durée = {ligne[1]}")
                            break
                self.listDateDouble[1].append(list(taches))
                taches.clear()
        for n in range(len(self.listDateDouble[0])):
            s = self.listDateDouble[1][n][0]
            pred = []
            for j in range(len(self.matriceAdj[s])):
                if self.matriceAdj[j][s] == 1:
                    pred.append(j)
            self.listDateDouble[2].append(list(pred))
            log(f"Pour le sommet {s}, prédécesseurs : {pred}")
        for i in range(len(self.listDateDouble[0])):
            if len(self.listDateDouble[2][i]) == 0:
                self.listDateDouble[3].append([0])
                log(f"Sommet {self.listDateDouble[1][i][0]} n'a pas de prédécesseur -> date = 0")
            else:
                valeurs = []
                for p in self.listDateDouble[2][i]:
                    for n in range(len(self.listDateDouble[1])):
                        if self.listDateDouble[1][n][0] == p:
                            calc = self.listDateDouble[1][n][1] + max(self.listDateDouble[3][n])
                            valeurs.append(calc)
                            log(f"Pour le sommet {self.listDateDouble[1][i][0]}, prédécesseur {p} -> date = {calc}")
                self.listDateDouble[3].append(list(valeurs))
        for tab in self.listDateDouble[3]:
            date = max(tab)
            self.listDateDouble[4].append(date)
        log(">> Calendrier au plus tôt calculé.")
        log()

    def auPlusTard(self):
        log(">> Calcul du calendrier au plus tard :")
        temp = [[], [], []]
        for i in range(len(self.matriceAdj)):
            succ = []
            s = self.listDateDouble[1][i][0]
            for j in range(len(self.matriceAdj)):
                if self.matriceAdj[s][j] == 1:
                    succ.append(j)
            temp[0].append(list(succ))
            log(f"Pour le sommet {s}, successeurs : {succ}")
        self.listDateDouble.append(temp[0])
        for _ in range(len(temp[0])):
            temp[1].append([])
        temp[1][len(temp[1]) - 1].append(self.listDateDouble[4][len(self.listDateDouble[0]) - 1])
        log(f"Pour le sommet de sortie, date au plus tard fixée à {self.listDateDouble[4][-1]}")
        valeur = []
        compteur = 2
        for i in range(len(self.listDateDouble[0]) - 2, -1, -1):
            successeurs = temp[0][i]
            for s in successeurs:
                for j in range(i, len(temp[0])):
                    if s == self.listDateDouble[1][j][0]:
                        calc = min(temp[1][j]) - self.listDateDouble[1][i][1]
                        valeur.append(calc)
                        log(f"Pour le sommet {self.listDateDouble[1][i][0]}, avec successeur {s} -> date = {calc}")
            for val in valeur:
                temp[1][len(temp[1]) - compteur].append(val)
            compteur += 1
            valeur.clear()
        self.listDateDouble.append(temp[1])
        temp[2] = [min(tab) for tab in temp[1]]
        self.listDateDouble.append(temp[2])
        log(">> Calendrier au plus tard calculé.")
        log()
        return self.listDateDouble

    def calculMarges(self):
        log(">> Calcul des marges :")
        listeMarge = []
        self.listComplete = copy.deepcopy(self.listDateDouble)
        for i in range(len(self.listComplete[0])):
            marge = self.listComplete[7][i] - self.listComplete[4][i]
            listeMarge.append(marge)
            log(f"Pour le sommet {self.listComplete[1][i][0]} : marge = {marge}")
        self.listComplete.append(listeMarge)
        log(">> Marges calculées.")
        log()

    def calculDesCheminsCritiques(self):
        for i in range(len(self.listComplete[0])):
            if self.listComplete[8][i] == 0:
                self.listCheminCritique.append(self.listComplete[1][i][0])
        log("Chemin(s) critique(s) : " + str(self.listCheminCritique))
        log()

    def affichageFinal(self):
        log(">> Affichage final du tableau récapitulatif :")
        tableau = PrettyTable()
        tab_field_names = ['Rangs', 'Tâches et sa longueur', 'Prédécesseurs', 'Dates par préd.',
                           'Date au plus tôt', 'Successeurs', 'Dates par succ.', 'Date au plus tard', 'Marge']
        tableau.field_names = tab_field_names
        lignes = list(zip(*self.listComplete))
        for ligne in lignes:
            tableau.add_row(ligne)
        log(tableau.get_string())


def main(file_number):
    global out_file
    out_file = open(f"Fichiers_Traces/trace {file_number}.txt", "w", encoding="utf-8")

    ascii_banner = pyfiglet.figlet_format("Bienvenue")
    log(ascii_banner)

    graphe = Graphe()
    log("Quel fichier voulez-vous analyser ? (Saisir -1 pour quitter)")

    try:
        fichierTest = "Fichiers_Tests/table " + str(file_number) + ".txt"
        log(f"\nLecture du fichier : {fichierTest}")
        graphe.readFile(fichierTest)

        graphe.matriceAdjacence()

        log("1. Affichage de la matrice des valeurs :")
        graphe.matriceValeur()

        log("2. Affichage du graphe comme un jeu de triplets :")
        graphe.affichageGrapheTriplets()

        log("3. Vérification des points d'entrée et de sortie, et détection de circuit :")
        if graphe.detectionCircuitEtPoints() == -1:
            return

        log("4. Calcul détaillé des rangs des sommets :")
        graphe.calculRang()

        log("5. Calcul du calendrier au plus tôt, au plus tard et des marges :")
        graphe.auPlusTot()
        graphe.auPlusTard()
        graphe.calculMarges()

        log("6. Calcul et affichage du(s) chemin(s) critique(s) :")
        graphe.calculDesCheminsCritiques()

        log("7. Affichage final des résultats :")
        graphe.affichageFinal()

    except FileNotFoundError:
        log("Ce fichier n'existe pas, veuillez réessayer !")
        log()

    out_file.close()


if __name__ == "__main__":
    for i in range(1,15):
        main(i)
