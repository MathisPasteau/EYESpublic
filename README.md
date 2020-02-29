# EYESpublic

J'ai exploré plusieurs pistes concernant la détection de feu piéton vert.

La fonction "crop" dans crosswalk.py pose encore quelques soucis avec certains fichiers. 
La fonction a pour but de crop une image sur les parties vertes (filtre HSV) de l'image en entrée.

je pense m'en servir pour constituer un dataset d'images (positiv and negativ) pour ensuite entrainer un fichier classifier.xml.
Le but final étant de reconnaitre le "petit bonhomme vert" dans la mesure du possible. 

J'ai peur que les tâches détectées soient trop petites et donc inexploitable.

