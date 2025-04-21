from Graphe import *
import pyfiglet

if __name__ == "__main__":

    ascii_banner = pyfiglet.figlet_format("Bienvenue")
    print(ascii_banner)

    numbFichier = 1
    while numbFichier != -1:

        graphe = Graphe()

        print("Quel fichier voulez-vous analyser ? (Saisir -1 pour quitter)")
        try:
            numbFichier = int(input())
        except ValueError:
            print("Saisie impossible, veuillez réessayer !")
            continue

        if numbFichier == -1:
            print("Au revoir !")
            break

        try:
            fichierTest = "Fichiers_Tests/table " + str(numbFichier) + ".txt"

            print(f"Lecture de : {fichierTest}")
            graphe.readFile(fichierTest)

            print("\t 1.Affichage du tableau de contrainte :")
            graphe.affichageTableau()

            print("\t 2.Affichage de la matrice d'adjacence :")
            graphe.matriceAdjacence()

            print("\t 3.Affichage de la matrice de valeur :")
            graphe.matriceValeur()

            graphe.affichageGraphique()

            print("\t 4.Verification d'arc et absence d'arc négatif  :")
            if graphe.verificationCircuitArcNegatif() == -1 :
                continue

            print("\t 5.Calcul du rang : ")
            graphe.calculRang()

            print("\t 6.Calcul dates : au plus tot et au plus tard + marge")
            graphe.auPlusTot()
            graphe.auPlusTard()
            graphe.calculMarges()
            graphe.affichageFinal()

            print("Chemin critique : ")
            graphe.calculDesCheminsCritiques()
            print(graphe.listCheminCritique)

            graphe.dureeMaxCheminCritique()
            graphe.affichageCheminCritique()

            print()

        except FileNotFoundError:
            print("Ce fichier n'existe pas, veuillez réessayer !")