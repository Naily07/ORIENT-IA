"""Génère des fiches RAG détaillées à partir du corpus structuré (DATA-1/DATA-3).

`backend/data/corpus.json` ne contenait que 20 fiches d'une phrase : le RAG ne
pouvait donc rien dire des matières, des débouchés ou de l'admission d'un
parcours — une question de suivi du type « et les matières de cette filière ? »
restait sans réponse alors que l'information existe, structurée, dans
`parcours.json` / `matieres.json` / `metiers.json`.

Ce script lit `CorpusFormations` et écrit `backend/data/corpus_genere.json` :
une fiche « matières », une fiche « débouchés » par parcours, une fiche par
mention, et un index par domaine (informatique, affaires, agronomie…). Les
fiches rédigées à la main dans `corpus.json` restent la référence et ne sont
jamais écrasées (`models.charger_corpus_rag()` les fait gagner en cas de
collision d'`id`).

**Traçabilité (§4).** Chaque fiche générée porte le `source_id` du modèle dont
elle est tirée (registre DATA-2) et, dans son texte, rappelle le statut de
cette source : les matières viennent des calendriers d'épreuves relayés par un
groupe étudiant (`SRC-CALENDRIERS-FACEBOOK`, statut *externe*), les débouchés
d'un guide généré automatiquement (`SRC-GUIDE-FILIERES-GENERE`, *externe*) —
à confirmer auprès de l'ISPM, jamais présentés comme officiels.

Lancement : `cd backend && python -m scripts.generer_corpus_rag`
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config import config
from src.models import (
    CorpusFormations,
    Mention,
    Parcours,
    charger_corpus,
    charger_corpus_formations,
)
from src.sources import statut_de_source

SORTIE = config.dossier_data / "corpus_genere.json"

# Catégories alignées sur celles déjà employées dans `corpus.json`, pour que le
# filtre par catégorie du RAG (`retrieve_context(categorie=…)`) reste cohérent.
CATEGORIE_PAR_MENTION = {
    "MENTION-INFO-TELECOM": "informatique_telecom",
    "MENTION-GENIE-INDUSTRIEL": "genie_industriel",
    "MENTION-GENIE-CIVIL-ARCHI": "genie_civil_architecture",
    "MENTION-DROIT-TECH-AFFAIRES": "droit_techniques_affaires",
    "MENTION-BIOTECH-AGRO": "biotechnologie_agronomie",
    "MENTION-TOURISME": "tourisme",
}

# Libellé lisible d'un domaine, pour les fiches d'index.
DOMAINE_LISIBLE = {
    "informatique_telecom": "informatique et télécommunications",
    "genie_industriel": "génie industriel",
    "genie_civil_architecture": "génie civil et architecture",
    "droit_techniques_affaires": "droit, commerce et techniques des affaires",
    "biotechnologie_agronomie": "biotechnologie et agronomie",
    "tourisme": "tourisme et hôtellerie",
}

# Mots-clés supplémentaires pour rendre une fiche d'index trouvable sur des
# formulations courantes qui n'emploient pas le vocabulaire exact de l'ISPM.
SYNONYMES_DOMAINE = {
    "informatique_telecom": "informatique, programmation, développement, réseaux, "
    "intelligence artificielle, data, télécommunications",
    "genie_industriel": "industrie, mécanique, électromécanique, chimie industrielle, "
    "mines, pétrole",
    "genie_civil_architecture": "bâtiment, construction, travaux publics, architecture, "
    "infrastructures, urbanisme",
    "droit_techniques_affaires": "commerce, gestion, management, entreprise, marketing, "
    "finance, comptabilité, droit des affaires, économie",
    "biotechnologie_agronomie": "agriculture, élevage, agronomie, agroalimentaire, "
    "pharmacie, plantes médicinales, environnement",
    "tourisme": "tourisme, hôtellerie, voyage, patrimoine, écotourisme",
}

STATUT_PHRASE = {
    "officiel": "Source officielle de l'ISPM.",
    "institutionnel": "Source institutionnelle.",
    "externe": (
        "Information de source externe (non publiée par l'ISPM) : à confirmer "
        "directement auprès de l'établissement."
    ),
}


def _phrase_statut(source_id: str | None) -> str:
    return STATUT_PHRASE.get(statut_de_source(source_id) or "", "Provenance non enregistrée.")


def _doc(
    doc_id: str,
    titre: str,
    categorie: str,
    contenu: str,
    source_id: str | None,
) -> dict:
    return {
        "id": doc_id,
        "titre": titre,
        "categorie": categorie,
        "contenu": " ".join(contenu.split()),
        "derniere_maj": datetime(2026, 8, 27).isoformat(),
        "source_id": source_id,
    }


def _noms_matieres(corpus: CorpusFormations, ids: list[str]) -> list[str]:
    par_id = {m.id: m.nom for m in corpus.matieres}
    vus: set[str] = set()
    noms: list[str] = []
    for identifiant in ids:
        nom = par_id.get(identifiant)
        if nom and nom.lower() not in vus:
            vus.add(nom.lower())
            noms.append(nom)
    return noms


def _noms_metiers(corpus: CorpusFormations, ids: list[str]) -> list[str]:
    par_id = {m.id: m.nom for m in corpus.metiers}
    return [par_id[i] for i in ids if i in par_id]


def _admission_de(corpus: CorpusFormations, parcours: Parcours) -> str | None:
    par_id = {p.id: p.description for p in corpus.prerequis}
    descriptions = [par_id[i] for i in parcours.prerequis if i in par_id]
    return descriptions[0] if descriptions else None


def _fiche_matieres(corpus: CorpusFormations, parcours: Parcours, categorie: str) -> dict | None:
    noms = _noms_matieres(corpus, parcours.matieres)
    if not noms:
        return None
    phrases = [
        f"Le parcours {parcours.nom} ({parcours.id}) fait étudier "
        f"{len(noms)} matières recensées."
    ]
    # Groupes de ~12 : chaque phrase reste un point d'ancrage d'embedding
    # distinct et répète le sigle du parcours, ce qui aide le retrieval.
    for i in range(0, len(noms), 12):
        tranche = ", ".join(noms[i : i + 12])
        phrases.append(f"Matières de {parcours.id} : {tranche}.")
    phrases.append(
        "La liste des matières provient des calendriers d'épreuves de l'ISPM "
        "relayés par un groupe étudiant ; " + _phrase_statut(None)
    )
    return _doc(
        f"DOC-{parcours.id}-MATIERES",
        f"Matières du parcours {parcours.nom} ({parcours.id})",
        categorie,
        " ".join(phrases),
        "SRC-CALENDRIERS-FACEBOOK",
    )


def _fiche_debouches(corpus: CorpusFormations, parcours: Parcours, categorie: str) -> dict | None:
    noms = _noms_metiers(corpus, parcours.debouches)
    if not noms:
        return None
    contenu = (
        f"Débouchés professionnels associés au parcours {parcours.nom} "
        f"({parcours.id}) : {', '.join(noms)}. "
        "Ces intitulés de métiers viennent d'un guide des filières généré "
        "automatiquement, non publié par l'ISPM : " + _phrase_statut("SRC-GUIDE-FILIERES-GENERE")
    )
    return _doc(
        f"DOC-{parcours.id}-DEBOUCHES",
        f"Débouchés du parcours {parcours.nom} ({parcours.id})",
        categorie,
        contenu,
        "SRC-GUIDE-FILIERES-GENERE",
    )


def _fiche_mention(
    corpus: CorpusFormations, mention: Mention, parcours: list[Parcours], categorie: str
) -> dict:
    sigles = ", ".join(f"{p.nom} ({p.id})" for p in parcours)
    admissions = {
        _admission_de(corpus, p) for p in parcours if _admission_de(corpus, p)
    }
    phrase_admission = (
        f" Admission : {' ; '.join(sorted(a for a in admissions if a))}." if admissions else ""
    )
    contenu = (
        f"La mention {mention.nom} de l'ISPM ({mention.niveau}) regroupe les "
        f"parcours suivants : {sigles}.{phrase_admission} "
        + _phrase_statut(mention.source_id)
    )
    return _doc(
        f"DOC-MENTION-{mention.id.replace('MENTION-', '')}",
        f"Mention {mention.nom} — parcours et admission",
        categorie,
        contenu,
        mention.source_id,
    )


def _fiche_domaine(categorie: str, parcours: list[Parcours]) -> dict:
    lisible = DOMAINE_LISIBLE.get(categorie, categorie)
    liste = ", ".join(f"{p.nom} ({p.id})" for p in parcours)
    contenu = (
        f"Filières de l'ISPM dans le domaine {lisible} : {liste}. "
        f"Mots-clés associés : {SYNONYMES_DOMAINE.get(categorie, '')}. "
        "Pour le détail des matières ou des débouchés d'un de ces parcours, "
        "voir sa fiche dédiée."
    )
    return _doc(
        f"DOC-DOMAINE-{categorie.upper()}",
        f"Quelles filières de l'ISPM en {lisible} ?",
        categorie,
        contenu,
        "SRC-ISPM-FILIERES",
    )


def generer() -> list[dict]:
    corpus = charger_corpus_formations()
    if not corpus.parcours:
        raise SystemExit(
            "Corpus structuré vide : lancer ce script depuis backend/ avec les "
            "données de backend/data présentes."
        )

    mention_par_id = {m.id: m for m in corpus.mentions}
    parcours_par_categorie: dict[str, list[Parcours]] = {}
    docs: list[dict] = []

    for parcours in corpus.parcours:
        categorie = CATEGORIE_PAR_MENTION.get(parcours.mention_id, "informations_generales")
        parcours_par_categorie.setdefault(categorie, []).append(parcours)
        for fiche in (
            _fiche_matieres(corpus, parcours, categorie),
            _fiche_debouches(corpus, parcours, categorie),
        ):
            if fiche:
                docs.append(fiche)

    for mention in corpus.mentions:
        categorie = CATEGORIE_PAR_MENTION.get(mention.id, "informations_generales")
        parcours = [p for p in corpus.parcours if p.mention_id == mention.id]
        if parcours:
            docs.append(_fiche_mention(corpus, mention, parcours, categorie))

    for categorie, parcours in sorted(parcours_par_categorie.items()):
        docs.append(_fiche_domaine(categorie, parcours))

    # Fiche transverse : la liste complète, pour « quelles sont toutes les
    # filières de l'ISPM ? »
    toutes = "; ".join(
        f"{p.nom} ({p.id}) — mention {mention_par_id[p.mention_id].nom}"
        for p in corpus.parcours
        if p.mention_id in mention_par_id
    )
    docs.append(
        _doc(
            "DOC-TOUTES-FILIERES",
            "Liste de toutes les filières (parcours) de l'ISPM",
            "informations_generales",
            f"L'ISPM propose {len(corpus.parcours)} parcours répartis dans "
            f"{len(corpus.mentions)} mentions : {toutes}.",
            "SRC-ISPM-FILIERES",
        )
    )
    return docs


def main() -> None:
    docs = generer()

    # Filet anti-collision : ne jamais produire un id déjà porté par une fiche
    # rédigée à la main (elles priment).
    ids_manuels = {d.id for d in charger_corpus("corpus.json")}
    collisions = sorted(d["id"] for d in docs if d["id"] in ids_manuels)
    if collisions:
        raise SystemExit(f"Collision d'id avec corpus.json : {collisions}")

    Path(SORTIE).write_text(
        json.dumps(docs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(docs)} fiches écrites dans {SORTIE.relative_to(config.dossier_data.parent.parent)}")


if __name__ == "__main__":
    main()
