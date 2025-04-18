import os
from CRN import generate_data
import torch
import numpy as np

def generer_donnees(nom_fichier, tmax):
    serie_temporelle = generate_data(nom_fichier, tmax)
    serie_temporelle = np.array(serie_temporelle)
    return torch.tensor(serie_temporelle).transpose(0, 1)

tmax = 10

# Liste des fichiers à traiter
fichiers = [
    "biomodels/BIOMD0000000001.xml",
    "biomodels/BIOMD0000000002.xml",
    "biomodels/BIOMD0000000082.xml",
    "biomodels/BIOMD0000000004.xml",
    "biomodels/BIOMD0000000086.xml",
    "biomodels/BIOMD0000000085.xml",
    "biomodels/BIOMD0000000014.xml",
    "biomodels/BIOMD0000000008.xml",
    "biomodels/BIOMD0000000011.xml",
    "biomodels/BIOMD0000000010.xml"
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
    serie_temporelle = generer_donnees(fichier, tmax)
    print(serie_temporelle)

    # Extraire l'ID du modèle en prenant uniquement les chiffres après 'BIOMD000000'
    model_id = fichier.split("/")[-1].split(".")[0].replace("BIOMD000000", "")
    
    # S'assurer que l'ID est formaté à 3 chiffres avec un zéro devant si nécessaire
    model_id = model_id.zfill(3)

    # Nom du fichier : features_XXX.pt
    nom_fichier_sortie = f"features/features_{model_id}.pt"

    # Enregistrement des caractéristiques
    torch.save(serie_temporelle, nom_fichier_sortie)
    print(f"Caractéristiques enregistrées dans {nom_fichier_sortie}")

# Affichage des fichiers enregistrés dans le répertoire "features"
print("\nListe des fichiers enregistrés dans le répertoire 'features' :")
for fichier in os.listdir("features"):
    if fichier.endswith(".pt"):
        print(f" - {fichier}")

