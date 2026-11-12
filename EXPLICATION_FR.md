# Explication simple du projet « Churn des joueurs (World of Warcraft) »

## 1. L'idée en une phrase

À partir des traces de connexion de vrais joueurs de *World of Warcraft* en 2008, on comprend
comment ils jouent, on les range en **groupes** (segments) et on prédit **qui va arrêter de
jouer** (le *churn*), pour donner des conseils à un studio de jeux.

## 2. Les mots clés

- **Churn** (attrition, abandon) : un joueur qui quitte le jeu. Exemple : un joueur qui ne se
  connecte plus pendant 60 jours.
- **Game analytics** : analyser les données des joueurs pour améliorer le jeu.
- **Avatar** : le personnage d'un joueur. Attention : un joueur peut avoir plusieurs avatars
  (appelés *alts*).
- **MMORPG** : jeu de rôle en ligne massivement multijoueur (des milliers de joueurs sur le même
  serveur).
- **Guilde** : un groupe de joueurs qui jouent ensemble.

## 3. Les données

- **WoW Avatar History** (Kaggle, licence **CC0** = domaine public, on peut tout réutiliser).
  Données collectées par des chercheurs taïwanais (Lee et al., 2011).
- Un programme (*crawler*) a noté **toutes les 10 minutes** qui était connecté sur un serveur.
- **10,8 millions de lignes**. Une ligne = « l'avatar 59425, niveau 1, Orc, Voleur, dans la
  zone Orgrimmar, guilde 165, était connecté le 01/01/2008 à 00:02 ».
- 37 354 avatars, toute l'année 2008, faction Horde.

## 4. Préparer les données : les sessions

Une **session** = une période de jeu continue. On regroupe les observations d'un même avatar
tant qu'il n'y a pas plus de 30 minutes d'écart.
Exemple : vu à 20:00, 20:10, 20:20, puis à 23:00 → 2 sessions (20:00–20:20, puis 23:00).
Durée d'une session = (dernière – première observation) + 10 minutes.
Résultat : **1 142 606 sessions**, durée médiane **60 minutes**.

On ajoute aussi le **type de zone** (ville, zone de quêtes, donjon, champ de bataille, arène)
grâce au fichier `zones.csv`. Petite décision : Shattrath, Silvermoon et Dalaran étaient
classées « Zone » mais ce sont des capitales → on les compte comme « City ».

## 5. Analyse exploratoire (EDA)

**EDA** = regarder les données avec des graphiques avant de faire des modèles.

- ≈ 1 900 avatars connectés par jour ; pic d'activité à **22h**.
- Gros pics de joueurs à la sortie du patch 3.0.2 (14 octobre 2008) et de l'extension
  *Wrath of the Lich King* (13 novembre 2008) → le nouveau contenu fait revenir les joueurs.
- **40,7 %** des avatars n'ont qu'**une seule** session dans l'année.
- Sur 18 885 avatars créés en 2008, seulement **14,4 %** atteignent le niveau 10 et **1,8 %**
  le niveau 70. Il faut en médiane **244 heures** de jeu pour arriver au niveau 70.
  → On perd la plupart des nouveaux personnages au tout début.

## 6. Définir le churn (étape la plus importante)

- On se place au **1er juillet 2008** (date « T »).
- On prend tous les avatars actifs dans les 30 jours avant T : **6 918 avatars**.
- **Churn = 1** si l'avatar n'est **jamais revu** pendant les **60 jours après T**. Sinon 0.
- Taux de churn : **29,1 %**.
- Les variables (*features*) utilisent **seulement les données avant T**. Sinon on tricherait en
  regardant le futur (c'est ce qu'on appelle une **fuite de données**, *data leakage*).

Exemples de variables :
- `recency_days` (récence) : nombre de jours depuis la dernière connexion. Ex : 12 jours.
- `active_days_30d` : nombre de jours où il a joué dans le dernier mois. Ex : 5.
- `tenure_days` (ancienneté) : depuis combien de jours on le connaît.
- `playtime_trend` : joue-t-il plus ou moins que le mois d'avant ? (positif = plus).
- `share_city`, `share_dungeon`, `share_pvp` : part du temps en ville, en donjon, en combat
  joueur contre joueur (*PvP*).
- `in_guild`, `guild_size` : est-il dans une guilde ? de quelle taille ?

Premiers constats :
- Niveau 1–20 : **52 %** de churn ; niveau 70 : **9 %**.
- Sans guilde : **47,6 %** de churn ; avec guilde : **14,5 %**.

## 7. Segmentation avec K-means

**K-means** : un algorithme qui range les joueurs en *k* groupes de joueurs qui se
ressemblent. Il place *k* « centres », met chaque joueur dans le groupe du centre le plus
proche, recalcule les centres, et recommence.

- Avant, on **standardise** les variables (même échelle) et on prend le **log** des variables très
  étalées (temps de jeu, taille de guilde).
- Pour choisir *k*, on utilise le **score de silhouette** : il mesure si chaque joueur est bien plus
  proche de son groupe que des autres groupes (entre −1 et 1, plus haut = mieux). On a testé
  k = 3 à 6 ; le meilleur est **k = 3**.
- **ACP / PCA** (figure 10) : on écrase les 9 variables en 2 dimensions juste pour dessiner les
  groupes sur un plan.

Les 3 segments :

| Segment | Nombre | Description | Churn |
|---|---:|---|---:|
| Joueurs engagés de fin de jeu | 2 680 | 129 h de jeu en 3 mois, niveau 67, donjons, grandes guildes | 8 % |
| Joueurs en montée de niveau (« questers ») | 2 811 | 9 h en 3 mois, niveau 22, 88 % du temps en quêtes | 48 % |
| Joueurs qui restent en ville (« city idlers ») | 1 427 | 2,4 h en 3 mois, sessions de 16 min, 93 % du temps en ville | 31 % |

## 8. Prédire le churn

- Découpage **75 % entraînement / 25 % test** (1 730 avatars), **stratifié** (même taux de churn
  des deux côtés).
- 3 modèles :
  - **Régression logistique** (modèle simple, linéaire) ;
  - **Random Forest** (300 arbres de décision qui votent) ;
  - **Gradient Boosting** (des arbres ajoutés un par un, chacun corrige les erreurs des
    précédents).
- `class_weight="balanced"` : on donne plus de poids aux churners, moins nombreux.

Métriques :
- **ROC-AUC** : probabilité qu'un churner pris au hasard ait un score plus élevé qu'un joueur fidèle.
- **PR-AUC** : qualité du compromis précision/rappel ; à comparer au taux de base (0,29).
- **Precision** : parmi les joueurs prédits « vont partir », combien partent vraiment.
- **Recall** : parmi ceux qui partent vraiment, combien on en détecte.
- **Validation croisée 5 plis** : on refait l'évaluation 5 fois sur 5 découpages différents.
- **Test « hors période »** (*out-of-time*) : on entraîne sur la photo du 1er mai et on teste sur
  celle du 1er juillet. C'est plus réaliste : dans la vraie vie, on prédit le futur avec le passé.

Résultats (vrais chiffres) :

| Modèle | ROC-AUC | F1 | Precision | Recall | Hors période |
|---|---:|---:|---:|---:|---:|
| Régression logistique | 0,868 | 0,674 | 0,573 | 0,817 | 0,862 |
| **Random Forest** | **0,885** | **0,697** | 0,651 | 0,750 | **0,880** |
| Gradient Boosting | 0,879 | 0,687 | 0,667 | 0,708 | 0,870 |

Le Random Forest est le meilleur, et il reste bon dans le test hors période (0,880).

## 9. Quelles variables comptent ?

- **Importance par permutation** : on mélange au hasard une colonne et on regarde combien le score
  baisse. Si ça baisse beaucoup, la variable est importante.
  → La **récence** est de loin la plus importante (−0,044 d'AUC), puis les jours actifs du dernier
  mois, l'ancienneté et le nombre de sessions.
- Attention : quand plusieurs variables disent la même chose (jours actifs, sessions, heures de
  jeu), elles se « partagent » l'importance.
- **Dépendance partielle** (*partial dependence*, figure 13) : comment la probabilité de churn
  prédite change quand une variable change. Exemple : elle monte très vite pendant les
  **7 à 10 premiers jours sans connexion**, puis se stabilise.

## 10. Recommandations pour un studio (à dire en entretien)

1. Relancer un joueur **dès 5–7 jours d'absence**, pas après un mois.
2. **Améliorer le début du jeu** : seulement 14 % des nouveaux personnages atteignent le niveau 10.
3. **Encourager le jeu social** (guildes) : les membres de guilde partent 3 fois moins.
   Mais c'est une **corrélation**, pas forcément une **cause** → à tester avec un test A/B.
4. Proposer des activités aux joueurs qui restent en ville.
5. Sortir du contenu régulièrement pour garder les joueurs engagés.
6. Utiliser le modèle comme un **score de risque hebdomadaire** et mesurer l'effet des actions.

**Test A/B** : on donne une nouveauté à la moitié des joueurs (groupe A) et pas à l'autre (groupe B),
puis on compare. C'est la seule façon de prouver qu'une action *cause* une amélioration.

## 11. Limites (à connaître)

- Avatar ≠ joueur (une personne peut avoir plusieurs personnages).
- Un seul serveur, une seule faction, en 2008.
- La définition du churn (60 jours) est un choix ; l'été peut l'influencer.
- Il y a des trous dans les données (pannes du crawler).
- K-means avec 3 groupes est une simplification.
