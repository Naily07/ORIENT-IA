"""Extraction des matières enseignées depuis l'archive de calendriers ISPM (DATA-1).

**Ce que c'est, et ce que ce n'est pas.** L'archive `backend/data/Matières.rar`
contient 129 images : pour chaque parcours et chaque niveau (L1 à M2), le
**calendrier des épreuves** d'une session. Chaque image porte l'en-tête de
l'ISPM, le tampon et la signature du Recteur. Ce ne sont donc **pas des
maquettes pédagogiques** : ce sont des plannings d'examens, dont on déduit les
matières effectivement évaluées cette session-là. Une matière enseignée mais
non évaluée dans la session photographiée n'y figure pas — c'est la limite
principale de cette source, et elle est enregistrée comme telle au registre
(§4).

**Provenance.** Les images proviennent d'un groupe Facebook d'étudiants de
l'ISPM, pas de l'établissement. Le document *se présente* comme officiel
(en-tête, tampon, signature), mais le canal d'acquisition ne permet ni d'en
vérifier l'authenticité, ni de garantir qu'il s'agit de la version courante.
Statut retenu au registre : **externe**. La règle non négociable du §4
s'applique — ces matières ne doivent pas être présentées comme une information
officielle de l'ISPM tant qu'elles n'ont pas été confirmées.

**Mécanisme reproductible** (livrable 3 du sujet). Ce script rejoue toute la
chaîne depuis l'archive :

    cd backend && python -m scripts.extraire_matieres --archive data/Matières.rar

Trois difficultés réelles de l'OCR, traitées ici plutôt que subies :

1. **Cellules multi-lignes fragmentées.** « Droit administratif » sort en deux
   blocs (« Droit », « administratif »). Les blocs sont donc recollés par
   proximité géométrique avant toute interprétation — d'où la conservation des
   boîtes englobantes.
2. **Espaces perdus.** « AnalyseMathématique » pour « Analyse Mathématique ».
   Les frontières minuscule→Majuscule sont rétablies, et le dédoublonnage se
   fait sur une clé canonique insensible aux espaces et aux accents, ce qui
   réunit les variantes d'un même libellé.
3. **En-têtes et pieds de page.** Nom de l'établissement, adresse, « Le
   Recteur », tampon, jours et dates : tout cela est du bruit structurel,
   filtré par des motifs explicites et testés
   (`backend/tests/test_extraction_matieres.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RACINE_DATA = Path(__file__).resolve().parent.parent / "data"

# --- Filtrage du bruit structurel ---------------------------------------------

_JOURS = r"(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)"

# `\b` est volontairement absent après le nom du jour : l'OCR colle la date au
# jour (« Jeudi05 », « LUNDI03.08.2026 ») et « i » suivi de « 0 » ne forme pas
# de frontière de mot, ce qui faisait passer tous ces blocs pour des matières.
MOTIFS_BRUIT: tuple[re.Pattern[str], ...] = (
    re.compile(r"^institut\s*sup[ée]rieur", re.I),
    re.compile(r"^ambatomaro", re.I),
    # `[i1l|]` : sur le tampon, l'OCR rend le « I » de I.S.P.M tantôt « 1 »,
    # tantôt « l ». Sans ces variantes, « 1.s.p.m » devenait une matière.
    re.compile(r"^[i1l|]\.?\s*s\.?\s*p\.?\s*m", re.I),
    # « rofesseur RABOANARY… » : l'OCR perd parfois la capitale initiale.
    # Le nom du Recteur est le marqueur fiable, pas le titre.
    re.compile(r"(?:p?rofesseur|raboanary|amedee|amédée)", re.I),
    re.compile(r"^le\s*recteur", re.I),
    re.compile(rf"^{_JOURS}", re.I),
    re.compile(r"^classe\s*:", re.I),
    re.compile(r"^semestre\b", re.I),
    re.compile(r"^(?:calendrier|planning|emploi\s*du\s*temps|examen)", re.I),
    # Groupes de classe isolés : « (INFO5) », « IMTICIA5-GIC5-BIO5) ».
    re.compile(r"^\(?(?:info|tour|ter|gic|bio|indus)\s*\d\)?$", re.I),
    # Code de parcours suivi d'une année, seul dans sa cellule : « Dtja5 ».
    re.compile(
        r"^\(?(?:dtja|caa|fic|emp|igglia|esiia|imticia|isaia|emii|icmp|gca|iaa|pip|aee|teh|tee)"
        r"\s*\d[a-z]?\)?$",
        re.I,
    ),
    re.compile(r"^\(?[A-Za-z]{2,10}\d\s*-\s*[A-Za-z]{2,10}\d"),
    # Fragments de l'en-tête et du logo découpés par l'OCR (« S.P.M », « TITUT »,
    # « POL »). Ils n'ont l'air de rien mais deviendraient des matières.
    re.compile(r"^(?:s\.?p\.?m|titut|polytec\w*|pol|ins?titut|madagascar|antananarivo)\b", re.I),
    # Mentions de modalité, pas des matières : « [TP pratique en ligne] ».
    re.compile(r"^\[.*\]$"),
    re.compile(r"^(?:tp\s+pratique|en\s+ligne|salle|amphi)\b", re.I),
    # Une date collée au libellé trahit une note de planning, pas une matière
    # (« TPTopographielejeudi10Novembre2022 »).
    re.compile(r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre"
               r"|octobre|novembre|d[ée]cembre)\s*\d{4}", re.I),
    # Ponctuation, chiffres ou dates seuls.
    re.compile(r"^[\W\d_]+$"),
)

# Suffixe indiquant à quel groupe d'étudiants l'épreuve s'adresse, pas une
# caractéristique de la matière : « CYBERSECURITE(INFO5) », « ANGLAIS(BIO5-INFO5) ».
_GROUPE_CLASSE = re.compile(
    r"\s*\(\s*(?:[A-Za-z]{2,10}\s*\d[A-Za-z]?)(?:\s*[-/]\s*[A-Za-z]{2,10}\s*\d[A-Za-z]?)*\s*\)\s*$"
)

# La parenthèse fermante manque souvent : la cellule est coupée au bord de
# l'image (« Microcontroleur(EMl15 », « Psychosociologie(ter5-dtja5 »). Sans
# ce second motif, ces libellés survivaient avec leur groupe accolé.
# Parenthèse jamais refermée en fin de libellé : la cellule a été coupée au
# bord de l'image. Un intitulé de matière légitime n'en comporte pas, donc la
# règle peut être large sans risquer d'amputer un vrai nom.
_GROUPE_CLASSE_OUVERT = re.compile(r"\s*\([^)]{0,30}$")

LONGUEUR_MINIMALE = 3


def _est_fragment_de_tampon(texte: str) -> bool:
    """Reste du tampon « I.S.P.M » mal lu (« T.s.p.m », « S.p.i », « 1.s.p »).

    Le tampon est circulaire : l'OCR en attrape des morceaux différents d'une
    image à l'autre, et chacun devenait une matière. Le test porte sur la
    composition — un sigle court fait uniquement des lettres de « ISPM » et
    contenant à la fois « s » et « p ». « Stat », « SNI », « PAO » et
    « Institutions Financieres » n'en font pas partie.
    """
    lettres = {c for c in texte.casefold() if c.isalpha()}
    return (
        len(texte) <= 8
        and lettres <= {"i", "s", "p", "m", "l", "t"}
        and {"s", "p"} <= lettres
    )


def est_bruit(texte: str) -> bool:
    """Vrai si le bloc relève de l'en-tête, du pied de page ou du calendrier."""
    nettoye = texte.strip()
    if len(nettoye) < LONGUEUR_MINIMALE:
        return True
    if nettoye.casefold().strip(". ") in FRAGMENTS_ENTETE:
        return True
    if _est_fragment_de_tampon(nettoye):
        return True
    return any(motif.search(nettoye) for motif in MOTIFS_BRUIT)


# --- Normalisation des libellés -----------------------------------------------

_MINUSCULE_MAJUSCULE = re.compile(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Ý])")


def retablir_espaces(texte: str) -> str:
    """Réinsère les espaces perdus par l'OCR aux frontières minuscule→Majuscule.

    « AnalyseMathématique » → « Analyse Mathématique ». Sans effet sur un
    libellé déjà correct, et sans effet non plus sur les mots tout en
    majuscules, où la frontière est indétectable.
    """
    return _MINUSCULE_MAJUSCULE.sub(" ", texte)


def retirer_groupe_classe(texte: str) -> str:
    """Retire le groupe d'étudiants accolé au libellé.

    « CYBERSECURITE(INFO5) » et « Cybersécurité » sont la même matière : le
    suffixe dit à quelle classe l'épreuve s'adresse, pas ce qui est enseigné.
    Le garder aurait démultiplié chaque matière de master en autant de
    variantes qu'il y a de groupes.
    """
    texte = _GROUPE_CLASSE.sub("", texte)
    return _GROUPE_CLASSE_OUVERT.sub("", texte).strip()


def _casse_lisible(texte: str) -> str:
    """Ramène un libellé tout en majuscules à une casse de titre.

    L'OCR rend « CYBERSECURITE » là où une autre image donne
    « Cybersécurité » : sans cette normalisation, l'affichage alternerait les
    deux selon le parcours. Les sigles courts (SNI, CDS, PHP, GAFI) sont
    laissés intacts — ce *sont* des majuscules.
    """
    if len(texte) > 4 and texte == texte.upper() and any(c.isalpha() for c in texte):
        return texte.capitalize()
    return texte


def _normaliser_parentheses(texte: str) -> str:
    """Ramène les parenthèses pleine chasse à leur équivalent ASCII.

    L'OCR rend parfois « （ » (U+FF08) au lieu de « ( ». Les motifs de
    nettoyage, écrits en ASCII, ne les voyaient pas : « Marketing（TER5-… »
    traversait toute la chaîne avec son groupe de classe accolé.
    """
    return texte.replace("（", "(").replace("）", ")")


def nettoyer_libelle(texte: str) -> str:
    """Libellé lisible : groupe de classe retiré, espaces normalisés, casse
    ramenée à une forme comparable."""
    texte = _normaliser_parentheses(re.sub(r"\s+", " ", texte).strip())
    texte = retirer_groupe_classe(texte)
    texte = retablir_espaces(texte).strip(" .,;:-–—•|()[]").strip()
    return _casse_lisible(texte)


# Fragments du bandeau « INSTITUT SUPERIEUR POLYTECHNIQUE » que l'OCR découpe
# en morceaux prononçables. Liste explicite et courte : filtrer tous les sigles
# de trois lettres supprimerait RDM, SNI, PAO ou CAE, qui sont de vraies
# matières.
FRAGMENTS_ENTETE = {"que", "quel", "titut", "pol", "spm", "ieur", "insti", "insti7", "tech"}

_MOTS_OUTILS = {
    "de", "des", "du", "d", "et", "en", "la", "le", "les", "l", "a", "au", "aux",
    "sur", "pour", "par", "un", "une", "dans",
}


def construire_lexique(libelles: list[str]) -> set[str]:
    """Vocabulaire tiré des libellés **déjà bien segmentés** du corpus.

    Le corpus est son propre dictionnaire : « Analyse Harmonique » apparaît
    correctement espacé sur certaines images, ce qui donne les mots
    nécessaires pour recouper « Analyseharmonique » ailleurs. Aucun
    dictionnaire français externe n'est requis, et le lexique reste cantonné au
    domaine — un mot courant hors sujet ne peut pas provoquer de découpe
    fantaisiste.
    """
    lexique = set(_MOTS_OUTILS)
    for libelle in libelles:
        for mot in re.split(r"[^\wà-ÿ']+", libelle.casefold()):
            mot = mot.strip("'")
            if len(mot) >= 3:
                lexique.add(mot)
    return lexique


def segmenter(texte: str, lexique: set[str], longueur_minimale: int = 13) -> str:
    """Rétablit les espaces d'un libellé que l'OCR a collé.

    Programmation dynamique : on cherche la découpe qui couvre toute la chaîne
    avec des mots du lexique, en préférant le moins de morceaux possible (donc
    les mots les plus longs). Si aucune découpe complète n'existe, le libellé
    est laissé tel quel — mieux vaut un mot collé qu'une découpe inventée.
    """
    if " " in texte or len(texte) < longueur_minimale:
        return texte

    minuscule = texte.casefold()
    n = len(minuscule)
    # meilleur[i] = (nombre de morceaux, index de début du dernier morceau)
    meilleur: list[tuple[float, int] | None] = [None] * (n + 1)
    meilleur[0] = (0.0, 0)
    for fin in range(1, n + 1):
        for debut in range(max(0, fin - 20), fin):
            if meilleur[debut] is None:
                continue
            morceau = minuscule[debut:fin].strip("'")
            if morceau not in lexique:
                continue
            cout = meilleur[debut][0] + 1
            if meilleur[fin] is None or cout < meilleur[fin][0]:
                meilleur[fin] = (cout, debut)

    if meilleur[n] is None:
        return texte

    coupes = []
    position = n
    while position > 0:
        debut = meilleur[position][1]
        coupes.append(texte[debut:position])
        position = debut
    return " ".join(reversed(coupes))


def cle_canonique(texte: str) -> str:
    """Clé de dédoublonnage insensible à la casse, aux accents et aux espaces.

    Indispensable ici : le même intitulé ressort selon les images en
    « Théorie des nombres », « Theorie des nombres » ou
    « Théoriedesnombres ». Sans cette clé, chacune deviendrait une matière
    distincte et le graphe compterait trois fois la même chose.
    """
    sans_accent = "".join(
        c
        for c in unicodedata.normalize("NFKD", texte.casefold())
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", sans_accent)


# --- Recollage des cellules multi-lignes ---------------------------------------


def _centre_vertical(boite) -> float:
    return sum(point[1] for point in boite) / len(boite)


def _etendue_horizontale(boite) -> tuple[float, float]:
    xs = [point[0] for point in boite]
    return min(xs), max(xs)


def recoller_cellules(blocs: list[dict], tolerance_verticale: float = 1.2) -> list[str]:
    """Recolle les blocs qui appartiennent visiblement à la même cellule.

    Deux blocs sont fusionnés s'ils se chevauchent horizontalement (même
    colonne) et sont verticalement contigus (moins d'une hauteur de ligne
    d'écart). C'est ce qui rend « Droit » + « administratif » à sa forme
    complète au lieu de produire deux matières fantômes.

    `blocs` : `[{"texte": str, "boite": [[x, y], ...]}]`, dans l'ordre de
    lecture rendu par l'OCR.
    """
    if not blocs:
        return []

    hauteurs = []
    for bloc in blocs:
        ys = [point[1] for point in bloc["boite"]]
        hauteurs.append(max(ys) - min(ys))
    hauteur_ligne = sorted(hauteurs)[len(hauteurs) // 2] or 1.0

    ordonnes = sorted(blocs, key=lambda b: (_centre_vertical(b["boite"])))
    groupes: list[list[dict]] = []
    for bloc in ordonnes:
        gauche, droite = _etendue_horizontale(bloc["boite"])
        centre = _centre_vertical(bloc["boite"])
        rattache = False
        for groupe in groupes:
            dernier = groupe[-1]
            g2, d2 = _etendue_horizontale(dernier["boite"])
            recouvre = min(droite, d2) - max(gauche, g2) > 0.4 * min(
                droite - gauche, d2 - g2
            )
            contigu = 0 < centre - _centre_vertical(dernier["boite"]) < (
                tolerance_verticale * hauteur_ligne
            )
            if recouvre and contigu:
                groupe.append(bloc)
                rattache = True
                break
        if not rattache:
            groupes.append([bloc])

    return [" ".join(b["texte"].strip() for b in groupe) for groupe in groupes]


# --- Extraction ----------------------------------------------------------------


def matieres_d_un_document(blocs: list[dict]) -> list[str]:
    """Libellés de matières d'une image, bruit retiré et cellules recollées."""
    libelles = []
    for texte in recoller_cellules(blocs):
        if est_bruit(texte):
            continue
        libelle = nettoyer_libelle(texte)
        if libelle and not est_bruit(libelle):
            libelles.append(libelle)
    return libelles


def agreger(documents: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    """Agrège les documents en un catalogue de matières et un programme.

    Retourne `(catalogue, programme)` :
    - `catalogue` : clé canonique → `{"nom", "occurrences", "variantes"}` ;
    - `programme` : une entrée par (parcours, niveau, semestre) avec ses
      matières, ce qui conserve l'information d'année que le modèle
      `Parcours.matieres` aplatit.
    """
    par_document = [matieres_d_un_document(d["blocs"]) for d in documents]
    # Le lexique est construit **après** un premier passage complet : il faut
    # avoir vu tous les libellés correctement espacés du corpus avant de
    # pouvoir recouper ceux que l'OCR a collés.
    lexique = construire_lexique([lib for libs in par_document for lib in libs])
    # Une seconde passe (réinjecter les mots récupérés au premier découpage dans
    # le lexique) a été essayée et mesurée : elle ne débloque aucun libellé
    # supplémentaire (106 avant, 106 après). Les libellés encore collés le sont
    # parce que leurs mots n'apparaissent nulle part séparément dans le corpus,
    # et aucune passe supplémentaire ne peut les inventer.

    variantes: dict[str, Counter] = defaultdict(Counter)
    programme: list[dict] = []

    for document, libelles in zip(documents, par_document, strict=True):
        # Nettoyage réappliqué après segmentation : le découpage peut faire
        # réapparaître en fin de libellé un groupe de classe que la première
        # passe ne voyait pas encore isolé. `nettoyer_libelle` est idempotent.
        libelles = [nettoyer_libelle(segmenter(lib, lexique)) for lib in libelles]
        cles = []
        for libelle in libelles:
            cle = cle_canonique(libelle)
            if not cle:
                continue
            variantes[cle][libelle] += 1
            if cle not in cles:
                cles.append(cle)
        programme.append(
            {
                "parcours": document["parcours"],
                "classe_imprimee": classe_imprimee(document["blocs"]),
                "parcours_concernes": list(parcours_concernes(document)),
                "niveau": document["niveau"],
                "semestre": document["semestre"],
                "fichier": document["fichier"],
                "matieres": cles,
            }
        )

    catalogue = {}
    for cle, compte in variantes.items():
        # Libellé retenu : le plus fréquent, en départageant à égalité par
        # celui qui contient des espaces (donc le mieux segmenté par l'OCR).
        nom = max(compte.items(), key=lambda kv: (kv[1], " " in kv[0], len(kv[0])))[0]
        catalogue[cle] = {
            "nom": nom,
            "occurrences": sum(compte.values()),
            "variantes": sorted(compte),
        }
    return catalogue, programme


# --- OCR ------------------------------------------------------------------------


def ocr_dossier(dossier: Path) -> list[dict]:
    """OCR de toutes les images, boîtes englobantes conservées.

    Les boîtes sont indispensables au recollage des cellules : sans elles, une
    cellule sur deux lignes produit deux matières incomplètes.
    """
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    moteur = RapidOCR()
    documents = []
    images = sorted(dossier.rglob("*.jp*g"))
    for i, chemin in enumerate(images, 1):
        relatif = chemin.relative_to(dossier)
        if len(relatif.parts) < 3:
            continue
        resultat, _ = moteur(np.array(Image.open(chemin).convert("RGB")))
        documents.append(
            {
                "niveau": relatif.parts[0],
                "parcours": relatif.parts[1],
                "semestre": relatif.stem,
                "fichier": str(relatif).replace("\\", "/"),
                "blocs": [
                    {"texte": texte, "boite": [[float(x), float(y)] for x, y in boite]}
                    for boite, texte, _ in (resultat or [])
                ],
            }
        )
        if i % 20 == 0:
            print(f"  OCR {i}/{len(images)}", file=sys.stderr)
    return documents


def extraire_archive(archive: Path, destination: Path) -> Path:
    """Décompresse l'archive RAR et retourne le dossier racine des images."""
    import subprocess

    unrar = Path(r"C:\Program Files\WinRAR\UnRAR.exe")
    if not unrar.exists():
        raise RuntimeError(
            f"UnRAR introuvable ({unrar}). Décompresser l'archive à la main et "
            "relancer avec --dossier."
        )
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(unrar), "x", "-idq", "-y", str(archive), str(destination) + "\\"],
        check=True,
    )
    racines = [c for c in destination.iterdir() if c.is_dir()]
    return racines[0] if len(racines) == 1 else destination


SOURCE_ID = "SRC-CALENDRIERS-FACEBOOK"

# Classes qui regroupent plusieurs parcours. L'ISPM mutualise les
# enseignements à certains niveaux : la classe imprimée sur le calendrier
# couvre alors toute une mention, pas un parcours isolé.
#
# - `BIO` : tronc commun L1–L2 de Biotechnologie et Agronomie, qui ne se
#   sépare en AEE/IAA/PIP qu'en L3.
# - `TOUR` : en M1 et M2, les deux parcours de la mention Tourisme sont réunis
#   en une seule classe. Le contenu du calendrier TOUR5 le confirme — il mêle
#   l'environnemental (Écotourisme, Écologie marine, Droit de l'environnement,
#   Gestion durable des ressources naturelles) et l'hôtelier (Art culinaire,
#   Civilisation appliquée au tourisme). Sans cette correspondance, TEE
#   n'aurait **aucune** matière, alors que la moitié de ce programme le
#   concerne directement.
CLASSES_REGROUPEES = {
    "BIO": ("AEE", "IAA", "PIP"),
    "TOUR": ("TEE", "TEH"),
}

_CLASSE_IMPRIMEE = re.compile(r"^classe\s*:\s*(.+)$", re.I)


def classe_imprimee(blocs: list[dict]) -> str | None:
    """Code de classe tel qu'imprimé sur le document, année retirée.

    **Pourquoi ne pas se fier au nom de dossier.** L'archive range les
    calendriers de master de la mention Tourisme sous `TEH`, alors que le
    document lui-même annonce « Classe : TOUR5 ». Le nom de dossier reflète le
    classement de la personne qui a archivé ; le libellé imprimé est la
    désignation de l'établissement. En cas de désaccord, c'est le document qui
    fait foi.
    """
    for bloc in blocs:
        trouve = _CLASSE_IMPRIMEE.match(bloc["texte"].strip())
        if trouve:
            code = re.sub(r"[^A-Za-z]", "", trouve.group(1)).upper()
            # L'OCR confond « I » et « l » sur ces en-têtes (« ESllA », « EMlI »).
            return code.replace("L", "I") if code else None
    return None


def parcours_concernes(document: dict) -> tuple[str, ...]:
    """Parcours auxquels rattacher les matières d'un document.

    Priorité au libellé imprimé, repli sur le nom de dossier, et prise en
    compte des classes qui regroupent plusieurs parcours.
    """
    imprime = classe_imprimee(document.get("blocs", []))
    for code, parcours in CLASSES_REGROUPEES.items():
        if imprime == code or document["parcours"].upper() == code:
            return parcours
    return (document["parcours"],)


def identifiant_matiere(cle: str) -> str:
    return f"MAT-{cle[:40].upper()}"


def ecrire_corpus(catalogue: dict, programme: list[dict], sortie: Path) -> dict:
    """Écrit `matieres.json` et `programme_matieres.json`, et met à jour
    `parcours.json` avec la relation parcours → matières.

    `Parcours.matieres` aplatit les cinq années en une liste unique, ce que le
    modèle actuel impose ; `programme_matieres.json` conserve à côté le détail
    par niveau et semestre, qui serait perdu autrement et que l'assistant peut
    exploiter plus tard (« quelles matières en L1 ? »).
    """
    matieres = [
        {
            "id": identifiant_matiere(cle),
            "nom": entree["nom"],
            "source_id": SOURCE_ID,
        }
        for cle, entree in sorted(catalogue.items(), key=lambda kv: kv[1]["nom"])
    ]
    (sortie / "matieres.json").write_text(
        json.dumps(matieres, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    par_parcours: dict[str, list[str]] = defaultdict(list)
    for entree in programme:
        cibles = entree.get("parcours_concernes") or (entree["parcours"],)
        for cible in cibles:
            for cle in entree["matieres"]:
                identifiant = identifiant_matiere(cle)
                if identifiant not in par_parcours[cible]:
                    par_parcours[cible].append(identifiant)

    chemin_parcours = sortie / "parcours.json"
    liste_parcours = json.loads(chemin_parcours.read_text(encoding="utf-8"))
    rattaches = 0
    for parcours in liste_parcours:
        trouvees = par_parcours.get(parcours["id"])
        if trouvees:
            parcours["matieres"] = trouvees
            rattaches += 1
    chemin_parcours.write_text(
        json.dumps(liste_parcours, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (sortie / "programme_matieres.json").write_text(
        json.dumps(
            [
                {
                    **entree,
                    "matieres": [identifiant_matiere(c) for c in entree["matieres"]],
                }
                for entree in programme
            ],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    sans_matiere = [p["id"] for p in liste_parcours if not p.get("matieres")]
    return {
        "matieres": len(matieres),
        "parcours_rattaches": rattaches,
        "parcours_sans_matiere": sans_matiere,
    }


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    source = parseur.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="archive RAR des calendriers")
    source.add_argument("--dossier", type=Path, help="dossier déjà décompressé")
    parseur.add_argument("--ocr-json", type=Path, help="réutiliser un OCR déjà calculé")
    parseur.add_argument(
        "--sortie", type=Path, default=RACINE_DATA, help="dossier de sortie des JSON"
    )
    parseur.add_argument(
        "--ecrire-corpus",
        action="store_true",
        help="écrit matieres.json/programme_matieres.json et met à jour parcours.json",
    )
    arguments = parseur.parse_args()

    if arguments.ocr_json and arguments.ocr_json.exists():
        documents = json.loads(arguments.ocr_json.read_text(encoding="utf-8"))
    else:
        dossier = arguments.dossier
        if arguments.archive:
            import tempfile

            dossier = extraire_archive(arguments.archive, Path(tempfile.mkdtemp()))
        documents = ocr_dossier(dossier)
        if arguments.ocr_json:
            arguments.ocr_json.write_text(
                json.dumps(documents, ensure_ascii=False), encoding="utf-8"
            )

    catalogue, programme = agreger(documents)
    print(f"{len(documents)} documents, {len(catalogue)} matières distinctes")

    (arguments.sortie / "matieres_brut.json").write_text(
        json.dumps({"catalogue": catalogue, "programme": programme}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    if arguments.ecrire_corpus:
        bilan = ecrire_corpus(catalogue, programme, arguments.sortie)
        print(
            f"  matieres.json : {bilan['matieres']} matières\n"
            f"  parcours.json : {bilan['parcours_rattaches']} parcours rattachés"
        )
        if bilan["parcours_sans_matiere"]:
            print(f"  sans matière  : {', '.join(bilan['parcours_sans_matiere'])}")
    print(f"Écrit dans {arguments.sortie}")


if __name__ == "__main__":
    main()
