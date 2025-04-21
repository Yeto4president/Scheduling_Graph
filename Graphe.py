import numpy as np
from prettytable import PrettyTable
import copy

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

    def readFile(self,chemin):
        with open(chemin, "r") as fichier:
            ligneFichier = fichier.readline()
            tableauContrainte = []
            while ligneFichier != "":
                ligneTableauContrainte = []
                compteur = 0
                contraintes = []
                i = 0
                while i < len(ligneFichier):
                    nombre = ""
                    while (i < len(ligneFichier) and ligneFichier[i] != " " and ligneFichier[i] != "\n"):
                        nombre += ligneFichier[i]
                        i += 1
                    if (nombre != "" and i < len(ligneFichier)):
                        compteur += 1
                        if (compteur >= 3):
                            contraintes.append(int(nombre))
                        else:
                            ligneTableauContrainte.append(int(nombre))
                    else:
                        i += 1
                ligneTableauContrainte.append(contraintes)
                tableauContrainte.append(ligneTableauContrainte)
                ligneFichier = fichier.readline()
            self.tableauContrainte = tableauContrainte

    def affichageTableau(self):
        x = PrettyTable()
        x.field_names = ["Tâche", "Durée", "Contraintes"]

        for i in range(0, len(self.tableauContrainte)):
            contraintes = ""
            for j in range(0, len(self.tableauContrainte[i][2])):
                contraintes += str(self.tableauContrainte[i][2][j]) + " "
            if (contraintes == ""):
                contraintes = "Aucun"
            x.add_row([self.tableauContrainte[i][0], self.tableauContrainte[i][1], contraintes])
        print(x)

    def matriceAdjacence(self):
        self.tableauContrainte = np.array(self.tableauContrainte, dtype=object)
        n = len(self.tableauContrainte)
        for i in range(n + 1):
            ligneMatrice = [0]
            terminaux = True
            for j in range(n):
                if i == 0:
                    terminaux = False
                    ligneMatrice.append(1 if not self.tableauContrainte[j][2] else 0)
                elif i in self.tableauContrainte[j][2]:
                    ligneMatrice.append(1)
                    terminaux = False
                else:
                    ligneMatrice.append(0)
            ligneMatrice.append(0 if not terminaux else 1)
            self.matriceAdj.append(ligneMatrice)
        self.matriceAdj.append([0] * (n + 2))
        self.matriceAdj = np.array(self.matriceAdj, dtype=int)

        tab_adj = PrettyTable()
        tab_field_names = ["\\", "α"] + [str(i) for i in range(1, len(self.matriceAdj) - 1)] + ["ω"]
        tab_adj.field_names = tab_field_names
        tab_tmp = np.array(["α"] + [str(i) for i in range(1, len(self.matriceAdj) - 1)] + ["ω"])
        matriceAdj = np.column_stack((tab_tmp, self.matriceAdj))
        for line in matriceAdj:
            tab_adj.add_row(line)
        print(tab_adj)

    def matriceValeur(self):
        self.tableauContrainte = np.array(self.tableauContrainte, dtype=object)
        for i in range(0, len(self.tableauContrainte) + 1):
            ligneMatrice = ['-']
            terminaux = True
            for j in range(0, len(self.tableauContrainte)):
                if (i == 0):
                    terminaux = False
                    if (len(self.tableauContrainte[j][2]) == 0):
                        ligneMatrice.append(0)
                    else:
                        ligneMatrice.append('-')
                elif (self.tableauContrainte[j][2].__contains__(i)):
                    ligneMatrice.append(self.tableauContrainte[i - 1][1])
                    terminaux = False
                else:
                    ligneMatrice.append('-')
            if (terminaux == False):
                ligneMatrice.append('-')
            else:
                ligneMatrice.append(self.tableauContrainte[i - 1][1])
            self.matriceVal.append(ligneMatrice)
        self.matriceVal.append(np.array([str('-')
                                   for i in range(len(self.tableauContrainte) + 2)]))

        tab_adj = PrettyTable()
        tab_field_names = ["\\", "α"] + [str(i) for i in range(1, len(self.matriceVal) - 1)] + ["ω"]
        tab_adj.field_names = tab_field_names
        tab_tmp = np.array(["α"] + [str(i) for i in range(1, len(self.matriceVal) - 1)] + ["ω"])
        matriceVal = np.column_stack((tab_tmp, self.matriceVal))
        for line in matriceVal:
            tab_adj.add_row(line)
        print(tab_adj)

    def verificationCircuitArcNegatif(self):
        def PointEntree(self):
            if self.matriceAdj is None:
                return False
            compteur_entree = 0

            for i in range(len(self.matriceAdj)):
                contient_un_1 = any(self.matriceAdj[j][i] == 1 for j in range(len(self.matriceAdj)))
                if not contient_un_1:
                    compteur_entree += 1
                    self.entree = i

            return compteur_entree == 1

        def PointSortie(self):
            if self.matriceAdj is None:
                return False
            compteur_sortie = 0

            for i in range(len(self.matriceAdj)):
                contient_un_1 = any(self.matriceAdj[i][j] == 1 for j in range(len(self.matriceAdj)))
                if not contient_un_1:
                    compteur_sortie += 1
                    self.sortie = i

            return compteur_sortie == 1

        def checkCircuit(self):
            self.matriceCopiee = np.copy(self.matriceAdj)
            for i in range(len(self.matriceCopiee)):
                tableau_pred = [j for j in range(len(self.matriceCopiee)) if self.matriceCopiee[j][i] == 1]
                tableau_succ = [j for j in range(len(self.matriceCopiee)) if self.matriceCopiee[i][j] == 1]
                for pred in tableau_pred:
                    for succ in tableau_succ:
                        self.matriceCopiee[pred][succ] = 1
            for i in range(len(self.matriceCopiee)):
                if self.matriceCopiee[i][i] == 1:
                    return True
            return False
        def arcIncidentIdentiques(self):
            taille = len(self.matriceVal)
            for i in range(1, taille):
                valeur = 0
                for j in range(taille):
                    if (self.matriceVal[i][j] != '-' and valeur == 0):
                        valeur = self.matriceVal[i][j]
                    elif (self.matriceVal[i][j] != valeur and self.matriceVal[i][j] != '-'):
                        return False
            return True

        def arcIncidentPointEntree(self):
            for i in range(1, len(self.matriceVal)):
                if (self.matriceVal[0][i] != 0 and self.matriceVal[0][i] != '-'):
                    return False
            return True

        def checkArcValeurNegative(self):
            if (None != self.matriceVal):
                for i in range(0, len(self.matriceVal)):
                    for j in range(len(self.matriceVal[i])):
                        if (self.matriceVal[i][j] != '-' and self.matriceVal[i][j] < 0):
                            return True
            return False

        print("Avec  α = 0  et ω = " + str(len(self.matriceVal) - 1))

        print("- Un seul point d'entrée ?")
        PointEntree(self)
        print(f"{self.entree} est une entrée")

        print("- Un seul point de sortie ?")
        PointSortie(self)
        print(f"{self.sortie} est une sortie")

        print("- Le graphe contient-il un circuit ? ")
        if checkCircuit(self):
            print("Il y a un CIRCUIT dans ce graphe ! L'ordonnancement n'est pas possible !")
            return -1
        else:
            print("OK, Le graphe ne contient pas de circuit ! ")

        print("\t- Arc incident indetiques ? ")
        if arcIncidentIdentiques(self):
            print("OK, Les valeurs des arcs incidents sont identiques ! ")

        else:
            print("ERREUR : valeurs NON IDENTIQUES pour tous les arcs incidents vers l’extérieur à un sommet")
            return -1

        print("\t- Arc incident entree nulle ? ")
        if arcIncidentPointEntree(self):
            print("OK, Valeurs null pour l'arc incident au point d'entrée ! ")
        else:
            print("ERREUR : valeurs NON NULL pour l'arc incident au point d'entree")
            return -1

        print("\t- Arc à valeur Négative ? ")
        if checkArcValeurNegative(self):
            print("ERREUR : arc NEGATIF")
            return -1
        else:
            print("OK ! ")
        return
    def calculRang(self):
        matriceCopie = np.copy(self.matriceAdj)
        etatRestant = len(self.matriceAdj)
        rang = 0

        while etatRestant > 0:
            eliminer = []
            print("Les états de RANG " + str(rang) + " sont : { ", end="")
            lignes = "Les etats de RANG " + str(rang) + " sont : { "
            for i in range(0, len(matriceCopie)):
                contientUn = False
                nbDeux = 0
                for j in range(0, len(matriceCopie)):
                    if (matriceCopie[j][i] == 2):
                        nbDeux += 1
                    if (matriceCopie[j][i] == 1):
                        contientUn = True
                        break

                if (contientUn == False and nbDeux != len(matriceCopie)):
                    eliminer.append(i)
                    etatRestant -= 1
            self.tableauRangs.append(list(eliminer))
            for m in range(0, len(eliminer)):
                if (eliminer[m] == 0):
                    print("α", end=" ")
                    lignes += "a "
                elif (eliminer[m] == len(self.matriceAdj) - 1):
                    print("ω", end=" ")
                    lignes += "w "
                else:
                    print(eliminer[m], end=" ")
                    lignes += str(eliminer[m]) + " "
                for n in range(0, len(matriceCopie)):
                    matriceCopie[eliminer[m]][n] = 2
                for n in range(0, len(matriceCopie)):
                    matriceCopie[n][eliminer[m]] = 2
            eliminer.clear()
            rang += 1
            print("}")
            lignes += "}\n"

    def auPlusTot(self):
        for i in range(0, len(self.tableauRangs)):
            for j in range(0, len(self.tableauRangs[i])):
                self.listDateDouble[0].append(i)
        for i in range(0, len(self.tableauRangs)):
            taches = []
            for k in range(0, len(self.tableauRangs[i])):
                taches.append(self.tableauRangs[i][k])
                if (self.tableauRangs[i][k] == 0 or self.tableauRangs[i][k] == len(self.matriceAdj) - 1):
                    taches.append(0)
                else:
                    for j in range(0, len(self.tableauContrainte)):
                        if (self.tableauContrainte[j][0] == (self.tableauRangs[i][k])):
                            taches.append(self.tableauContrainte[j][1])
                            break
                self.listDateDouble[1].append(list(taches))
                taches.clear()

        for n in range(len(self.listDateDouble[0])):
            i = self.listDateDouble[1][n][0]
            etatPrec = []
            for j in range(len(self.matriceAdj[i])):
                if (self.matriceAdj[j][i] == 1):
                    etatPrec.append(j)
            self.listDateDouble[2].append(list(etatPrec))
            etatPrec.clear()

        for i in range(len(self.listDateDouble[0])):

            if (len(self.listDateDouble[2][i]) == 0):
                self.listDateDouble[3].append([0])
            else:
                valeur = []
                for j in range(len(self.listDateDouble[2][i])):
                    val = 0
                    for n in range(len(self.listDateDouble[1])):
                        if (self.listDateDouble[1][n][0] == self.listDateDouble[2][i][j]):
                            valeur.append(
                                self.listDateDouble[1][n][1] + max(self.listDateDouble[3][n]))
                self.listDateDouble[3].append(list(valeur))
                valeur.clear()
        for tab in self.listDateDouble[3]:
            self.listDateDouble[4].append(max(tab))

    def auPlusTard(self):
        temp = [[], [], []]
        for loop in range(len(self.matriceAdj)):
            successeurs = []
            i = self.listDateDouble[1][loop][0]
            for j in range(len(self.matriceAdj)):
                if (self.matriceAdj[i][j] == 1):
                    successeurs.append(j)
            temp[0].append(list(successeurs))
            successeurs.clear()
        self.listDateDouble.append(temp[0])

        for l in range(len(temp[0])):
            temp[1].append([])

        temp[1][len(temp[1]) - 1].append(self.listDateDouble[4][len(self.listDateDouble[0]) - 1])
        valeur = []
        compteur = 2
        for i in range(len(self.listDateDouble[0]) - 2, -1, -1):
            successeurs = temp[0][i]

            for k in range(len(successeurs)):
                for j in range(i, len(temp[0])):
                    if (successeurs[k] == self.listDateDouble[1][j][0]):
                        valeur.append(
                            min(temp[1][j]) - self.listDateDouble[1][i][1])

            for val in valeur:
                temp[1][len(
                    temp[1]) - compteur].append(val)
            compteur += 1
            valeur.clear()

        self.listDateDouble.append(temp[1])
        for tab in temp[1]:
            temp[2].append(min(tab))
        self.listDateDouble.append(temp[2])
        return self.listDateDouble

    def calculMarges(self):
        listeMarge = []
        self.listComplete = copy.deepcopy(self.listDateDouble)
        for i in range(len(self.listComplete[0])):
            listeMarge.append(self.listComplete[7][i] - self.listComplete[4][i])
        self.listComplete.append(listeMarge)

    def affichageFinal(self):
        tableau = PrettyTable()

        tab_field_names = ['Tâches, sa longueur']
        for i in range(0, len(self.listComplete[0])):
            tab_field_names.append(self.listComplete[1][i])
        tableau.field_names = tab_field_names

        tab_tmp = np.array(['Rangs', 'Tâches et sa longueur', 'Predecesseur', 'Date par Pred.', 'Date au plus tôt', 'Successeurs', 'Date par Succ.', 'Date au plus tard', 'Marge'])
        self.listComplete = np.array(self.listComplete, dtype=object)
        self.listComplete = np.column_stack((tab_tmp, self.listComplete))

        comp = 0
        for line in self.listComplete:
            if (comp != 1):
                tableau.add_row(line)
            comp += 1
        print(tableau)

    def calculDesCheminsCritiques(self):
        for i in range(len(self.listComplete[0])):
            if (self.listComplete[8][i] == 0):
                self.listCheminCritique.append(self.listComplete[1][i][0])

    def affichageGraphique(self, layout='spring', show_edge_labels=True):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        n = self.matriceAdj.shape[0]

        labels = {}
        for i in range(n):
            if i == 0:
                labels[i] = "α"
            elif i == n - 1:
                labels[i] = "ω"
            else:
                labels[i] = str(i)
            G.add_node(i)

        for i in range(n):
            for j in range(n):
                if self.matriceAdj[i, j] == 1:
                    G.add_edge(i, j)

        plt.figure(figsize=(14, 10))

        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'rang':
            pos = {}
            for r, liste_sommets in enumerate(self.tableauRangs):
                for index, sommet in enumerate(liste_sommets):
                    pos[sommet] = (index * 3, -r * 3)
            for node in G.nodes():
                if node not in pos:
                    pos[node] = (0, 0)
        else:
            pos = nx.spring_layout(G, seed=42, k=3, iterations=100)

        nx.draw_networkx_nodes(
            G, pos,
            node_size=1500,
            node_color="skyblue",
            edgecolors='black'
        )

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=12,
            font_color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )

        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowsize=25,
            arrowstyle='-|>',
            edge_color='black',
            width=2,
            connectionstyle='arc3,rad=0.1',
            min_source_margin=20,
            min_target_margin=20
        )

        if show_edge_labels and self.matriceVal is not None:
            edge_labels = {}
            for i in range(len(self.matriceVal)):
                for j in range(len(self.matriceVal[i])):
                    val = self.matriceVal[i][j]
                    if val != '-' and val != '0' and val != 0:
                        if G.has_edge(i, j):
                            edge_labels[(i, j)] = str(val)

            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_color='red',
                font_size=10,
                label_pos=0.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

        plt.axis('off')
        plt.title("Représentation Graphique du Graphe", fontsize=16)
        plt.tight_layout()
        plt.show()

    def dureeMaxCheminCritique(self):
        try:
            if not self.listCheminCritique:
                print("Aucun chemin critique trouvé.")
                return 0

            duree_max = 0
            critical_path = self.listCheminCritique

            for i in range(len(critical_path)):
                node = critical_path[i]

                if node != 0 and node != len(self.matriceAdj) - 1:
                    for task in self.tableauContrainte:
                        if task[0] == node:
                            duree_max += task[1]
                            break

            print(f"Durée maximale du chemin critique : {duree_max}")
            return duree_max

        except Exception as e:
            print(f"Erreur inattendue : {e}")
            return None

    def affichageCheminCritique(self, layout='spring'):
        import networkx as nx
        import matplotlib.pyplot as plt

        critical_path = self.listCheminCritique

        G = nx.DiGraph()

        for node in critical_path:
            G.add_node(node)

        for i in range(len(critical_path) - 1):
            u = critical_path[i]
            v = critical_path[i + 1]
            if self.matriceAdj[u, v] == 1:
                G.add_edge(u, v)

        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=3, iterations=100)

        labels = {}
        n_total = self.matriceAdj.shape[0]
        for node in G.nodes():
            if node == 0:
                labels[node] = "α"
            elif node == n_total - 1:
                labels[node] = "ω"
            else:
                labels[node] = str(node)

        nx.draw_networkx_nodes(
            G, pos,
            node_color="red",
            node_size=1500,
            edgecolors='black'
        )

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=12,
            font_color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )

        nx.draw_networkx_edges(
            G, pos,
            edge_color='red',
            arrows=True,
            arrowsize=25,
            arrowstyle='-|>',
            width=4,
            connectionstyle='arc3,rad=0.1',
            min_source_margin=20,
            min_target_margin=20
        )

        plt.title("Graphe du Chemin Critique", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
