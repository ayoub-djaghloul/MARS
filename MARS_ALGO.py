import random
import time #Certaines des bibliothèques ne sont pas nécessaires
import csv
import os
import concurrent.futures
import sys
import numpy as np
from read_write import read_word_vectors
from ranking import *
from prettytable import PrettyTable

output_csv_file = 'resultats_tableau.csv'

def generationmasolutionaleatoire(length):
    max_ones = (length // 2) - 1
    num_ones = random.randint(1, max_ones)  # Nombre de 1 dans la solution, en garantissant au moins un True mais pas tout pour respecter la condition donnée
    solution = [True] * num_ones + [False] * (length - num_ones)  # Initialisation de la solution avec le bon nombre de 1 et 0
    random.shuffle(solution)  # Mélange de la solution pour l'aléatoire
    return solution

def filtrer_embeddings_par_solution(embeddings, solution):
    # Filtrage de chaque embedding pour ne garder que les dimensions où la solution est True
    embeddings_filtrés = [[dim for dim, keep in zip(embed, solution) if keep] for embed in embeddings]
    return embeddings_filtrés



def generer_nom_fichier(file_path, prefix="embeddings_filtrés", extension=".txt"):
    base_name = os.path.basename(file_path)  # Extrait "result_30_MC.txt"
    base_name_without_ext = os.path.splitext(base_name)[0]  # Retire l'extension pour avoir "result_30_MC"
    i = 1
    while True:
        new_file_name = f"{prefix}({base_name_without_ext})({i}){extension}"
        # Vérifie si le fichier existe déjà
        if not os.path.exists(new_file_name):
            break
        i += 1
    return new_file_name

#Version séquentielle de ma génération de voisins   
def generationvoisin(solution, l):
    unique_neighbors = set()  # Ensemble pour stocker des représentations uniques des voisins
    neighbors = []  # Liste pour stocker les voisins sous forme de listes

    while len(neighbors) < l:
        # Copie de la solution
        neighbor = solution[:]  
        
        # Sélection aléatoire d'un index
        rand_index = random.randint(0, len(solution) - 1)  
        
        # Inversion de l'élément sélectionné
        neighbor[rand_index] = not neighbor[rand_index]  
        
        # Vérification si le voisin contient au moins un True et un False
        while sum(neighbor) == 0 or sum(neighbor) == len(neighbor):
            neighbor[rand_index] = not neighbor[rand_index]  # Annuler la dernière inversion
            rand_index = random.randint(0, len(solution) - 1)  
            neighbor[rand_index] = not neighbor[rand_index]
        
        # Vérification du nombre de 1 dans le voisin
        while sum(neighbor) > (len(neighbor) // 2) - 1:
            neighbor[rand_index] = not neighbor[rand_index]  # Annuler la dernière inversion
            rand_index = random.randint(0, len(solution) - 1)  
            neighbor[rand_index] = not neighbor[rand_index]
        
        # Convertir le voisin en chaîne pour vérifier l'unicité
        neighbor_str = ''.join(['1' if x else '0' for x in neighbor])
        if neighbor_str not in unique_neighbors:
            unique_neighbors.add(neighbor_str)
            neighbors.append(neighbor)
    
    print("Voisins générés:")
    for i, neighbor in enumerate(neighbors, start=1):
        print(f"v{i}: {' '.join(map(str, neighbor))}")        
    return neighbors

def generation_voisin_aleatoire(solution, existing_neighbors):
    #Génère un seul voisin aléatoire unique à partir de la solution donnée.
    max_ones = (len(solution) // 2) - 1  # Le nombre maximal de 1 autorisés
    attempts = 0

    while attempts < 100:  # Limite les tentatives pour éviter une boucle infinie
        neighbor = solution[:]  # Copie de la solution initiale
        rand_index = random.randint(0, len(solution) - 1)  # Sélection aléatoire d'un index
        neighbor[rand_index] = not neighbor[rand_index]  # Inversion de la valeur à cet index

        # Convertir la liste en tuple pour pouvoir l'ajouter à un ensemble
        neighbor_tuple = tuple(neighbor)

        if 0 < sum(neighbor) <= max_ones and neighbor_tuple not in existing_neighbors:
            existing_neighbors.add(neighbor_tuple)  # Ajouter le nouveau voisin unique à l'ensemble
            print("Voisin généré:", " ".join(map(str, neighbor)))
            return neighbor  # Retourner le voisin unique

        attempts += 1

    raise ValueError("Impossible de générer un voisin unique après plusieurs tentatives.")

def parallel_fitness_with_borne(voisins, embeddings, beta, alpha):
    max_fitness = None
    min_fitness = None
    fitness_results = []

    # Utilisation de ThreadPoolExecutor pour calculer la fitness en parallèle
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fitness, voisin, embeddings, beta) for voisin in voisins]
        for future in concurrent.futures.as_completed(futures):
            fitness_result = future.result()
            fitness_results.append(fitness_result)

            # Mise à jour des valeurs maximale et minimale
            if max_fitness is None or fitness_result > max_fitness:
                max_fitness = fitness_result
            if min_fitness is None or fitness_result < min_fitness:
                min_fitness = fitness_result

    # Calcul de la borne d'inclusion
    borne_inclusion = max_fitness - alpha * (max_fitness - min_fitness)

    return fitness_results, max_fitness, min_fitness, borne_inclusion


def parallel_generation_voisins(solution, l):
    #Génère des voisins en parallèle en utilisant ThreadPoolExecutor, en évitant les doublons.
    existing_neighbors = set()  # Ensemble pour stocker les voisins existants

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generation_voisin_aleatoire, solution, existing_neighbors) for _ in range(l)]
        neighbors = [future.result() for future in concurrent.futures.as_completed(futures)]
    return neighbors

def fitness(v, embeddings, beta):
    n = len(v)
    Lv = sum(v)  # Nombre de 1 dans v
    m = len(embeddings)  # Nombre d'embeddings
    matrix=constructionmatricediscernabilite(embeddings,epsilon)
    Cv = calculdeCv(v, matrix)  # Nombre de combinaisons d'objets que v peut discerner
    term1 = beta * (n - Lv) / n
    term2 = (1 - beta) * (Cv / ((m ** 2 - m) / 2)) 
   
    return term1 + term2



def calculdeCv(v, matrix):
    Cv = 0
    for j in range(len(matrix[0])):
        # Initialiser la somme de la colonne à zéro
        col_total = 0

        # Parcourir chaque ligne de la matrice
        for i in range(len(matrix)):

            if v[j]:
                col_total += matrix[i][j]

        # Ajouter la somme de la colonne à la somme totale
        Cv+= col_total

    return Cv



def constructionmatricediscernabilite(embeddings, epsilon):
    num_embeddings = len(embeddings)
    matrix = []

    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            row = []
            for dim1, dim2 in zip(embeddings[i], embeddings[j]):
                c=round(abs(dim2 - dim1),2)
                if c <= epsilon:
                    #Prendre uniquement les deux dixièmes de la fonction Python
                   
                    row.append(0)
                else:
                    row.append(1)
            matrix.append(row)
    return matrix


def calculborne(voisins,embeddings,beta):
    max_fitness = 0
    min_fitness = 1000000
   
    for i, voisin in enumerate(voisins, start=1):
    # Calculer la fitness du voisin
        fitness_voisin = fitness(voisin, embeddings, beta)
    # Afficher la fitness
        #print(f"Fitness de v{i}:", fitness_voisin)
    # Mettre à jour le plus grand et le plus petit fitness
        max_fitness = max(max_fitness, fitness_voisin)
        min_fitness = min(min_fitness, fitness_voisin)
    borne_inclusion=max_fitness-alpha*(max_fitness-min_fitness)
    print("Plus grand fitness parmi les voisins:", max_fitness)
    print("Plus petit fitness parmi les voisins:", min_fitness)
    print("Ma borne d'inclusion b= ",borne_inclusion)
    return borne_inclusion  



def voisinborne(embeddings,voisins,borne,beta):
    voisins_excedant_borne = []
    for i, voisin in enumerate(voisins, start=1):
         fitness_voisin = fitness(voisin, embeddings, beta)
         if fitness_voisin > borne:
            voisins_excedant_borne.append(voisin)
    return voisins_excedant_borne
   
def construire_listerestreinte_parallele(voisins, embeddings, beta, borne_inclusion):
    #Construit la liste restreinte en parallèle.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Calculer la fitness en parallèle pour chaque voisin
        future_to_voisin = {executor.submit(fitness, voisin, embeddings, beta): voisin for voisin in voisins}

        # Filtrer les voisins basés sur la borne d'inclusion
        voisins_satisfaisants = []
        for future in concurrent.futures.as_completed(future_to_voisin):
            voisin = future_to_voisin[future]
            fitness_voisin = future.result()
            if fitness_voisin > borne_inclusion:
                voisins_satisfaisants.append(voisin)

    # Affichage des voisins satisfaisants pour le débogage
    print("LRC = {", end="")
    for i, voisin_satisfaisant in enumerate(voisins_satisfaisants, start=1):
        voisin_str = " ".join(["1" if v else "0" for v in voisin_satisfaisant])
        print(f"v{i}: {voisin_str}", end=", ")
    print("}")

    return voisins_satisfaisants
   
def construirelisterestreinte(voisins, borne_inclusion):
    voisins_satisfaisants = []
    print("LRC = {", end="")
    first = True
    for i, voisin in enumerate(voisins, start=1):
        if fitness(voisin, numeric_embeddings, beta) > borne_inclusion:
            voisins_satisfaisants.append(voisin)  # Ajouter le voisin satisfaisant à la liste
            voisin_numerique = [v for v in voisin if isinstance(v, (int, float))]
            if not first:
                print(",", end="")
            print(f"v{i}:", voisin_numerique, end="")
            first = False
    print("}")
    return voisins_satisfaisants

def load_embeddings(file_path):
    words = []
    embeddings = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            words.append(parts[0])  # Conserver le mot vu que l'autre groupe se plaint de ne pas l'avoir dans les résultats
            embeddings.append([float(value) for value in parts[1:]])  # Conserver l'embedding
    return words, embeddings


def associer_mots_embeddings(words, embeddings_filtrés):
    return [(word, embed) for word, embed in zip(words, embeddings_filtrés)]





def GRASP(embeddings, epsilon=0.1, beta=0.1, alpha=0.1, niter=100, l=1, initial_solution=None):
    if initial_solution is None:
        initial_solution = generationmasolutionaleatoire(len(embeddings[0]))
    stop_after = 20 # Nombre d'itérations sans amélioration après lesquelles arrêter
    iter_since_last_improvement = 0  # Compteur d'itérations depuis la dernière amélioration
    print("Remise à 0 car un changement a été effectué sur la solution optimale.Valeur actuelle :")
    print(iter_since_last_improvement)
    best_solution = initial_solution
    best_fitness = fitness(initial_solution, embeddings, beta)

    iter = 0
    while iter < niter and iter_since_last_improvement < stop_after:
        iter += 1
        voisinsgeneres = parallel_generation_voisins(best_solution, l)
        fitness_results, max_fitness, min_fitness, borne = parallel_fitness_with_borne(voisinsgeneres, embeddings, beta, alpha)
        
        listerestreinte = construirelisterestreinte(voisinsgeneres, borne)
        if listerestreinte:
            sol1 = random.choice(listerestreinte)
            current_fitness = fitness(sol1, embeddings, beta)
            if current_fitness >= best_fitness:
                print("La solution choisie au hasard est meilleure.Je fais alors un switch")
                print(current_fitness," >= ",best_fitness)
                best_solution = sol1
                best_fitness = current_fitness
                iter_since_last_improvement = 0  # Amélioration
                print("Remise à 0 car un changement a été effectué sur la solution optimale.Valeur actuelle :",iter_since_last_improvement)
        
            else:
                print("La solution choisie au hasard n'est meilleure.On garde alors la solution initiale")
                print(current_fitness," < ",best_fitness)
                iter_since_last_improvement += 1  # Aucune amélioration
                print("Incrémentation car aucun changement n'a été effectué sur la solution optimale.Valeur actuelle :",iter_since_last_improvement)
        else:
            # Si aucune solution dans la liste restreinte n'améliore la solution actuelle,dans le sens ici si le fitness est moindre
            iter_since_last_improvement += 1
            print("Incrémentation car aucun changement n'a été effectué sur la solution optimale.Valeur actuelle :",iter_since_last_improvement)
    print(f"D'après mon implémentation, la meilleure solution après {iter} itérations semble être {best_solution}.")
    embeddings_filtrés = filtrer_embeddings_par_solution(embeddings, best_solution)
    embeddings_avec_mots = associer_mots_embeddings(words, embeddings_filtrés)  

    return best_solution, embeddings_avec_mots


if __name__ == "__main__":
    start_time = time.time()

    file_path = sys.argv[1]
    alpha = 0.4  # Définir la valeur d'alpha souhaitée
    beta = 0.5   # Définir la valeur de beta souhaitée
    l = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    nbiter = int(sys.argv[4])
    word_vecs = []
                      
    # Chargement des embeddings
    words, numeric_embeddings = load_embeddings(file_path)
    print("Embeddings de départ: ")
    print(numeric_embeddings)
    # Taille de la solution initiale basée sur la première entrée d'embedding
    n = len(numeric_embeddings[0])

    # Génération d'une solution initiale aléatoire
    initial_solution = generationmasolutionaleatoire(n)
    # Vérification du nombre d'arguments passés
    if len(sys.argv) != 5:
        print("Utilisation de cet algorithme: python grasp_junio(1).py <chemin_embeddings> <l> <epsilon> <nbiter>")
        sys.exit(1)

    # Exécution de GRASP
    solution, embeddings_filtrés = GRASP(numeric_embeddings, epsilon, beta, alpha, nbiter, l, initial_solution)
    temps_ecoule = time.time() - start_time

    # Affichage des résultats
    meilleur_fitness = fitness(solution, numeric_embeddings, beta)
    print(f"Meilleure solution: {solution}, Fitness: {meilleur_fitness}, Temps écoulé: {temps_ecoule}")
    nouveau_fichier_embeddings = generer_nom_fichier(file_path)

    # Enregistrement des résultats dans un fichier CSV (facultatif)
    dimensionsup = n - sum(solution)
    file_exists = os.path.isfile('GRASP_Junior(version finale).csv')
    should_write_header = not os.path.exists('GRASP_Junior(version finale).csv') or os.stat('GRASP_Junior(version finale).csv').st_size == 0

    with open('GRASP_Junior(version finale).csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if should_write_header:
         writer.writerow(['Fichier_Embeddings_Filtrés', 'Solution_optimale', 'Meilleur_fitness', 'Dimensions_depart', 'Nombre d\'éléments', 'Nombre_voisins', 'Nombre_itérations', 'epsilon', 'beta', 'alpha', 'Temps d\'exécution', 'Dimensions_supprimées'])
        writer.writerow([nouveau_fichier_embeddings, solution, meilleur_fitness, n, len(numeric_embeddings), l, nbiter, epsilon, beta, alpha, temps_ecoule, dimensionsup])

    # Écriture des embeddings filtrés dans le nouveau fichier
    with open(nouveau_fichier_embeddings, 'w') as f:
        for word, embed in embeddings_filtrés: 
            ligne = word + ' ' + ' '.join(map(str, embed))
            f.write(ligne + '\n')

    print(f"Les embeddings filtrés ont été sauvegardés dans {nouveau_fichier_embeddings}")
    wordvec = read_word_vectors(nouveau_fichier_embeddings)
    word_vecs.append(wordvec)



word_sim_dir = "word-sim"
alpha = 0.4  # Définir la valeur d'alpha souhaitée
beta = 0.5   # Définir la valeur de beta souhaitée

table = PrettyTable()
table.field_names = ["Serial", "Dataset-Name", "Num Pairs (WP)"] +[f"file avec {alpha} et {beta}"] 

for i, filename in enumerate(os.listdir(word_sim_dir)):
    manual_dict = {}
    rho_values = []
    total_size = 0
    for word_vec in word_vecs:
        auto_dict = {}
        
        for line in open(os.path.join(word_sim_dir, filename), 'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in word_vec and word2 in word_vec:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(word_vec[word1], word_vec[word2])
            total_size += 1
        rho_value = spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)) * 100
        rho_values.append("{:.4f}".format(rho_value))
    
    # Ajouter une ligne au tableau
    row_values = [ 1, filename, total_size] + rho_values
    table.add_row(row_values)
    # Afficher le tableau
    print(table)

    # Écriture des résultats dans le fichier CSV
with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(table.field_names)  # Écriture de l'en-tête du tableau
        writer.writerow(row_values)

print(f"Les résultats du tableau ont été enregistrés dans {output_csv_file}")
