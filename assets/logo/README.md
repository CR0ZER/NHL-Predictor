# NHL Predictor — kit logo (piste 01 « Sigmoïde »)

La courbe représente la fonction logistique du modèle, et le point au bout représente le palet.

## Arborescence
| Dossier | Contenu | Usage |
|---|---|---|
| `svg/` | Tous les logos en vectoriel, avec le texte vectorisé (aucune police requise) | Web, Figma, source pour tous les autres formats |
| `pdf/` | Versions horizontales et empilées en PDF vectoriel | Impression, documents, envoi à un imprimeur |
| `png/horizontal`, `png/stacked` | Logos détourés (fond transparent), largeurs 512, 1024 et 2048 px | Slides, README, réseaux |
| `png/mark` | Le symbole seul, détouré, de 128 à 1024 px | Avatars, filigranes |
| `png/app-icon` | Icône carrée à coins arrondis, fond navy, de 256 à 1024 px | Avatar GitHub, Discord, Slack |
| `web/` | favicon.ico (16/32/48), favicon.svg, apple-touch-icon, android-chrome, icône maskable, site.webmanifest, snippet `<head>` | Site GitHub Pages |
| `social/` | Image Open Graph 1200×630 et aperçu social GitHub 1280×640 | Partage de liens, paramètres du repo |

## Variantes
- `color-dark` : à poser sur un fond foncé (idéalement #0E2236).
- `color-light` : à poser sur un fond clair (blanc ou #F2F6F9).
- `mono-navy`, `mono-black`, `mono-white` : une seule couleur, pour la gravure, le tampon, les fonds photo ou le N&B.

Les versions en couleur gardent les 3 lignes de grille. Dans les versions mono et dans les icônes, je les ai retirées pour que ça reste lisible en petit.

## Couleurs
| Rôle | Hex |
|---|---|
| Navy (fond / texte) | `#0E2236` |
| Glace (fond clair) | `#F2F6F9` |
| Accent cyan (palet, « PREDICTOR » sur fond foncé) | `#22B8F0` |
| Accent foncé (« PREDICTOR » sur fond clair, pour le contraste) | `#0B6E99` |

## Typographie
Barlow Condensed ExtraBold Italic (Google Fonts, licence SIL OFL 1.1, usage commercial autorisé). Le texte est vectorisé dans tous les fichiers.

## Règles rapides
- Zone de protection autour du logo : au moins la hauteur du palet × 2.
- Taille mini : 24 px de haut pour le logo horizontal. En dessous, utiliser l'icône seule.
- Ne pas déformer, ne pas changer les couleurs, ne pas mettre la version `color-light` sur un fond foncé.

## Intégration sur GitHub Pages
1. Copier le contenu de `web/` et `social/og-image-1200x630.png` à la racine publiée.
2. Coller `web/head-snippet.html` dans le `<head>` de `ui.html`.
3. GitHub › Settings › Social preview : y mettre `social/github-social-preview-1280x640.png`.
