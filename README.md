# Analyse et Modélisation de Séries Temporelles Énergétiques - Données Enedis

## Contexte du Projet

Dans le cadre de notre projet d'analyse de données et d'intelligence artificielle appliqué au secteur de l'énergie : 
L'étude se base sur les données en open data d'Enedis, spécifiquement le jeu de données RES2-6-9kVA (clients résidentiels avec chauffage électrique, puissance souscrite entre 6 et 9 kVA). 

L'objectif principal est de concevoir un tableau de bord analytique complet permettant d'étudier les comportements de consommation électrique (valeurs exprimées en puissance kW au pas de 30 minutes, converties en énergie kWh), de classifier les profils de clients, de prédire la consommation future et de générer des profils synthétiques.

## Objectifs et Méthodologie

Le projet est structuré autour de quatre axes majeurs d'apprentissage et de développement :

1. **Clustering (Apprentissage non supervisé)**
   - **Objectif :** Distinguer les Résidences Principales (RP) des Résidences Secondaires (RS).
   - **Méthode :** Création de variables explicatives (feature engineering) à partir des séries temporelles, réduction de dimensionnalité éventuelle (ACP, décomposition de Fourier) et application de l'algorithme K-Means. Le clustering nous permet d'étiqueter nos données pour la phase de classification.

2. **Classification (Apprentissage supervisé)**
   - **Objectif :** Prédire le type de profil (RP ou RS) à partir des comportements de consommation.
   - **Méthode :** Entraînement et comparaison de plusieurs modèles (Régression Logistique, Random Forest). L'évaluation s'appuie sur la précision (Accuracy) et la matrice de confusion, en veillant à l'équilibrage des classes dans le jeu de test.

3. **Forecasting (Prévision)**
   - **Objectif :** Prédire la consommation journalière ou les pics de charge futurs.
   - **Méthode :** Implémentation de modèles de régression (Régression Linéaire, Random Forest Regressor).

4. **Génération de données**
   - **Objectif :** Développer un générateur de courbes de charge conditionné par le type de client (RP ou RS).
   - **Méthode :** Synthèse de données et évaluation de la cohérence globale par rapport aux distributions réelles observées.

## Architecture du Code

Pour répondre aux bonnes pratiques de développement logiciel et garantir la maintenabilité du code, l'application a été modularisée :

- `main.py` : Script principal agissant comme chef d'orchestre. Il initialise l'application Streamlit et fait le lien entre le traitement des données et l'affichage.
- `backend.py` : Module dédié à la logique métier. Il contient les fonctions de chargement, de transformation des données (conversion kW vers kWh), d'ingénierie des caractéristiques et l'entraînement des modèles d'Intelligence Artificielle.
- `frontend.py` : Module dédié à la conception de l'interface utilisateur. Il gère la configuration de la page, l'intégration des graphiques Plotly et la mise en page du tableau de bord.

## Prérequis et Installation

### 1. Clonage du dépôt

Clonez ce dépôt sur votre machine locale :

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DOSSIER>
```

### 2. Environnement virtuel et dépendances
Il est fortement recommandé de travailler dans un environnement virtuel pour isoler les dépendances du projet.

```bash
# Création de l'environnement virtuel
python -m venv env

# Activation de l'environnement (Windows)
env\Scripts\activate
# Activation de l'environnement (Linux/MacOS)
source env/bin/activate

# Installation des bibliothèques requises
pip install -r requirements.txt
```

### 3. Gestion du jeu de données
Attention : En raison de sa taille, le fichier de données brut dataset.csv n'est pas versionné sur Git.

Pour exécuter le code, vous devez :

Télécharger le jeu de données Enedis (RES2-6-9kVA) : https://drive.google.com/drive/folders/1iM6HijTZJ5KSlhz2tZz6_Yzy5yFizPzp?usp=drive_link

Nommer ce fichier export.csv

Placer ce fichier à la racine du projet, dans le même répertoire que main.py.

Le chemin d'accès est géré de manière dynamique via la bibliothèque os pour garantir la compatibilité sur n'importe quel système d'exploitation.

## Utilisation
Pour lancer l'application interactive (Dashboard), exécutez la commande suivante à la racine du projet : 

```bash
streamlit run main.py
```

Une fenêtre s'ouvrira automatiquement dans votre navigateur par défaut, vous permettant de naviguer à travers les différentes analyses de profils, les métriques des modèles d'IA et les simulations.
