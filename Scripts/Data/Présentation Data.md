## Dataset

Le dataset initial `investments_VC.csv` contient 54 294 startups multi-sectorielles. Après filtrage pharma/biotech :

| Statut | Nombre |
|--------|--------|
| Total secteur pharma/biotech | 4 207 |
| Acquisitions (succès) | 225 |
| Fermetures (échec) | 147 |
| Encore en activité (operating) | 3 730 |

Colonnes principales utilisées :

- `category_list` et `market` : classification sectorielle
- `funding_total_usd` : montant total levé en USD
- `funding_rounds` : nombre de tours de financement
- `founded_at` : date de création
- `status` : statut final (acquired / closed / operating)
- `post_ipo_equity` et `post_ipo_debt` : détection des IPO non étiquetées

Sources textuelles pour l'analyse éthique : publications PubMed, sites web des entreprises, essais cliniques ClinicalTrials.gov
