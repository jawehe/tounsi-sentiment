## Dataset: TuniziDataset (Kaggle)
- Source: [colle ici le lien Kaggle réel]
- Licence: [à vérifier sur la page Kaggle, section "License"]
- Nombre d'exemples: 96 476 (version binaire, après nettoyage) / ~100 000 (brut, 3 classes)
- Labels: -1 (négatif), 0 (neutre), 1 (positif)
- Langue: arabizi tunisien (issu de Twitter)
- Format: CSV (colonnes: InputText, SentimentLabel)
- Qualité: texte bruité typique des réseaux sociaux, mais exploitable après nettoyage (suppression doublons, URLs, mentions, normalisation des répétitions de lettres)
- Distribution des classes: 52.7% positif, 43.8% négatif, 3.5% neutre (fort déséquilibre sur le neutre)
- Décision: utilisé comme base principale. Version binaire (positif/négatif) retenue pour le modèle de production (81% accuracy). Version 3 classes explorée en parallèle, limitée par le déséquilibre du neutre (voir notebook 03).