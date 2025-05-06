import libsbml
from CRNTools.CRN import *
from CRNTools.solver_call import *
import os
import numpy as np
import torch


solver = Kvaerno5()
key = random.key(0)

a = -10
b = -4 
c = 5
tmax = 10000 
nb_trajectoires = 5

def generer_donnees(nom_fichier):
    crn = CRNGraph(libsbml.readSBML(nom_fichier).getModel())
    odes, y0, p0, _ = crn.generateODEs()


    series, nkey = computeTrajectoriesUntilSteadyState(odes, y0, p0, tmax, solver, a,b,c,key=key, n_trajectories=nb_trajectoires)

    resultat = []
    for temps, valeurs in series:
        ts = torch.tensor(np.array(temps))
        vals = torch.tensor(np.array(valeurs))
        vals = vals.transpose(0,1)
        resultat.append(vals)
        #resultat.append((ts,vals))
    return resultat, nkey

# Liste des fichiers à traiter
fichiers = [
    "biomodels/BIOMD0000000001.xml",
    "biomodels/BIOMD0000000002.xml",
    "biomodels/BIOMD0000000050.xml",
    "biomodels/BIOMD0000000060.xml",
    "biomodels/BIOMD0000000080.xml",
    "biomodels/BIOMD0000000085.xml",
    "biomodels/BIOMD0000000086.xml",
    "biomodels/BIOMD0000000087.xml",
    "biomodels/BIOMD0000000040.xml",
    "biomodels/BIOMD0000000017.xml"
]

# Création du dossier de sortie si besoin
os.makedirs("features", exist_ok=True)
# Suppression des fichiers existants dans le répertoire 'features' pour éviter les doublons
for fichier in os.listdir("features"):
    if fichier.endswith(".pt"):
        os.remove(os.path.join("features", fichier))
# Boucle sur chaque fichier
for fichier in fichiers:
    print(f"\nTraitement du fichier : {fichier}")
    resultat, nouvelle_clef = generer_donnees(fichier)
    key = nouvelle_clef
    print(resultat)
    # Extraire l'ID du modèle en prenant uniquement les chiffres après 'BIOMD000000'
    model_id = fichier.split("/")[-1].split(".")[0].replace("BIOMD000000", "")
    model_id = model_id.zfill(3)
    # Nom du fichier : features_XXX.pt
    nom_fichier_sortie = f"features/features_{model_id}.pt"
    # Enregistrement des caractéristiques
    torch.save(resultat, nom_fichier_sortie)
    print(f"Caractéristiques et le temps sont enregistrées dans {nom_fichier_sortie}")

# Affichage des fichiers enregistrés dans le répertoire "features"
print("\nListe des fichiers enregistrés dans le répertoire 'features' :")
for fichier in os.listdir("features"):
    if fichier.endswith(".pt"):
        print(f"- {fichier}")





#resultat, nouvelle_clef = generer_donnees("biomodels/BIOMD0000000005.xml")
#key = nouvelle_clef
