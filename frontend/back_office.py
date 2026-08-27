"""Back-office — l'espace d'administration et de contrôle.

Destiné à l'équipe et au jury, pas au candidat. Il répond à une question que le
front-office ne peut pas poser : **est-ce que ce système est fiable, et
comment le sait-on ?**

Cinq vues, chacune adossée à un artefact réellement produit par le dépôt plutôt
qu'à une affirmation :

| Vue | Adossée à |
|---|---|
| Tableau de bord | `GET /health`, corpus structuré, configuration calibrée |
| Observabilité | `GET /observabilite/traces` (§15) |
| Qualité des données | `graphe.detecter_incoherences` (ONTO-4), registre des sources (§4) |
| Corpus & graphe | corpus structuré + graphe de connaissances (ONTO-2) |
| Mesures | `eval_results_ml.json`, `eval_results_rag_calibration.json`, `eval_results.json` |

Quand un artefact manque, la vue affiche **la commande qui le produit** plutôt
qu'un panneau vide : un tableau de bord qui ne dit pas pourquoi il est vide
pousse à conclure que le système ne mesure rien.
"""

from __future__ import annotations

from collections import Counter

import streamlit as st

from noyau import (
    BACKEND_IMPORTABLE,
    ApiIndisponible,
    afficher_mention_obligatoire,
    api_get,
    artefact_absent,
    charger_json_local,
    exiger_acces_admin,
    sante,
)


def _exiger_backend() -> bool:
    if not BACKEND_IMPORTABLE:
        st.error(
            "Cette vue lit le corpus en local et nécessite que le paquet `src` soit "
            "importable. Voir l'avertissement en haut de page.",
            icon="⛔",
        )
        return False
    return True


# --- Tableau de bord ----------------------------------------------------------


def page_tableau_de_bord() -> None:
    exiger_acces_admin()
    st.title("🛠️ Tableau de bord")
    afficher_mention_obligatoire(compacte=True)

    charge = sante()
    if charge is None:
        st.error("API injoignable — les indicateurs ci-dessous sont indisponibles.", icon="⛔")
    else:
        corpus = charge.get("corpus", {})
        colonnes = st.columns(4)
        colonnes[0].metric("Mentions", corpus.get("mentions", 0), border=True)
        colonnes[1].metric("Parcours", corpus.get("parcours", 0), border=True)
        colonnes[2].metric(
            "Clé LLM", "configurée" if charge.get("cle_llm_configuree") else "absente", border=True
        )
        colonnes[3].metric("Modèle", charge.get("modele", "?"), border=True)

    st.markdown("#### Configuration de recherche documentaire")
    st.caption(
        "Valeurs **calibrées** sur le corpus ISPM (RAG-5), pas héritées. Voir l'onglet "
        "Mesures pour le balayage complet."
    )
    if _exiger_backend():
        from src.config import config

        colonnes = st.columns(4)
        colonnes[0].metric("Seuil de pertinence", config.rag_seuil_pertinence, border=True)
        colonnes[1].metric("k (passages)", config.rag_k, border=True)
        colonnes[2].metric("Itérations agent max", config.agent_max_iterations, border=True)
        colonnes[3].metric(
            "Seuil de confiance", config.orchestrateur_seuil_confiance, border=True
        )

    st.markdown("#### État d'avancement des données")
    st.caption(
        "Le corpus est incomplet et c'est documenté : matières, compétences et "
        "débouchés n'ont pas de source fiable identifiée (BACKLOG.md, DATA-1). "
        "L'afficher évite qu'une démonstration laisse croire le contraire."
    )
    if _exiger_backend():
        from src.models import charger_corpus_formations

        corpus_structure = charger_corpus_formations()
        colonnes = st.columns(4)
        colonnes[0].metric("Matières", len(corpus_structure.matieres), border=True)
        colonnes[1].metric("Compétences", len(corpus_structure.competences), border=True)
        colonnes[2].metric("Métiers", len(corpus_structure.metiers), border=True)
        colonnes[3].metric("Prérequis", len(corpus_structure.prerequis), border=True)


# --- Observabilité ------------------------------------------------------------


def page_observabilite() -> None:
    exiger_acces_admin()
    st.title("📊 Observabilité")
    st.caption(
        "Traces du pipeline (§15 du sujet) : question, contexte, décision, outils, "
        "latence — telles qu'écrites par le backend, sans retraitement."
    )

    limite = st.slider("Nombre de traces", 5, 200, 30)
    try:
        traces = api_get("/observabilite/traces", limite=limite)
    except ApiIndisponible as erreur:
        st.error(f"Traces indisponibles : {erreur}", icon="⛔")
        return

    if not isinstance(traces, list) or not traces:
        st.info("Aucune trace enregistrée — posez d'abord une question dans le front-office.")
        return

    latences = [t.get("latence_ms", 0) or 0 for t in traces]
    actions = Counter((t.get("decision") or {}).get("action", "?") for t in traces)
    outils = Counter(
        outil
        for t in traces
        for outil in ((t.get("decision") or {}).get("outils_utilises") or [])
    )

    colonnes = st.columns(4)
    colonnes[0].metric("Traces", len(traces), border=True)
    colonnes[1].metric("Latence moyenne", f"{round(sum(latences) / len(latences))} ms", border=True)
    colonnes[2].metric("Latence max", f"{max(latences)} ms", border=True)
    colonnes[3].metric("Actions distinctes", len(actions), border=True)

    gauche, droite = st.columns(2)
    with gauche:
        st.markdown("**Répartition des actions**")
        st.bar_chart(dict(actions), horizontal=True, height=240)
    with droite:
        st.markdown("**Outils appelés**")
        if outils:
            st.bar_chart(dict(outils), horizontal=True, height=240)
        else:
            st.caption("Aucun outil appelé sur cet échantillon.")

    st.markdown("**Latence par requête (ms)**")
    st.line_chart(list(reversed(latences)), height=200)

    filtre = st.multiselect("Filtrer par action", sorted(actions), default=[])
    st.markdown("#### Détail des traces")
    for trace in traces:
        decision = trace.get("decision") or {}
        action = decision.get("action", "?")
        if filtre and action not in filtre:
            continue
        titre = (
            f"{str(trace.get('horodatage', '?'))[:19]} · {action} · "
            f"{trace.get('latence_ms', '?')} ms"
        )
        with st.expander(titre):
            st.json(trace)


# --- Qualité des données ------------------------------------------------------


def page_qualite_donnees() -> None:
    exiger_acces_admin()
    st.title("🔎 Qualité et traçabilité des données")
    st.caption(
        "Contrôles déterministes, sans LLM : cohérence structurelle du corpus (ONTO-4) "
        "et provenance de chaque information (§4 du sujet)."
    )
    if not _exiger_backend():
        return

    from src.graphe import construire_graphe, detecter_incoherences
    from src.models import charger_corpus_formations
    from src.sources import charger_registre_sources, verifier_provenance

    corpus = charger_corpus_formations()
    graphe = construire_graphe(corpus)
    incoherences = detecter_incoherences(corpus, graphe)
    manquantes = [i for i in incoherences if i.get("donnee_manquante")]
    contradictions = [i for i in incoherences if not i.get("donnee_manquante")]

    colonnes = st.columns(3)
    colonnes[0].metric("Constats", len(incoherences), border=True)
    colonnes[1].metric("Données non collectées", len(manquantes), border=True)
    colonnes[2].metric("Contradictions réelles", len(contradictions), border=True)

    st.info(
        "**La distinction est portée par l'outil lui-même**, pas par cette interface. "
        "Une donnée pas encore collectée (DATA-1) n'est pas un défaut de fiabilité du "
        "corpus — les confondre transformerait un chantier en alerte.",
        icon="ℹ️",
    )

    if contradictions:
        st.markdown("#### ⚠️ Contradictions à traiter")
        st.dataframe(contradictions, width="stretch")
    else:
        st.success("Aucune contradiction structurelle détectée dans le corpus.", icon="✅")

    if manquantes:
        with st.expander(f"Données non encore collectées ({len(manquantes)})"):
            st.dataframe(manquantes, width="stretch")

    st.markdown("#### Registre des sources (§4)")
    registre = charger_registre_sources()
    if not registre:
        st.warning("Registre des sources vide.")
        return

    statuts = Counter(entree.statut for entree in registre)
    colonnes = st.columns(len(statuts) or 1)
    for colonne, (statut, nombre) in zip(colonnes, sorted(statuts.items()), strict=False):
        colonne.metric(statut.capitalize(), nombre, border=True)

    st.dataframe(
        [
            {
                "id": e.id,
                "titre": e.titre,
                "statut": e.statut,
                "consultée le": str(e.date_consultation),
                "limites connues": len(e.limites),
                "url": e.url,
            }
            for e in registre
        ],
        width="stretch",
    )

    orphelines = verifier_provenance(
        [m.source_id for m in corpus.mentions] + [p.source_id for p in corpus.parcours],
        registre,
    )
    if orphelines:
        st.error(f"Références de source orphelines : {orphelines}", icon="⛔")
    else:
        st.success(
            "Toute donnée du corpus qui déclare une source pointe vers une entrée "
            "réelle du registre.",
            icon="✅",
        )

    with st.expander("Limites déclarées par source — à lire avant toute citation"):
        for entree in registre:
            if entree.limites:
                st.markdown(f"**{entree.id}** ({entree.statut})")
                for limite in entree.limites:
                    st.markdown(f"- {limite}")


# --- Corpus et graphe ---------------------------------------------------------

_COULEURS_NOEUDS = {
    "Parcours": "#1f6f5c",
    "Mention": "#5b4b8a",
    "Prerequis": "#a85a25",
    "Competence": "#2c6fa8",
    "Metier": "#7a2c5c",
    "Matiere": "#3d6b1f",
}


def _echapper_dot(texte: str) -> str:
    """Échappe une étiquette DOT.

    Un nom contenant un guillemet casserait le graphe entier — le corpus actuel
    n'en contient pas, mais rien ne le garantit une fois les matières et
    compétences collectées (DATA-1).
    """
    return texte.replace("\\", "\\\\").replace('"', '\\"')


def _graphe_dot(types_retenus: set[str]) -> str:
    from src.graphe import construire_graphe, type_et_id
    from src.models import charger_corpus_formations

    graphe = construire_graphe(charger_corpus_formations())
    lignes = ["digraph ontologie {", "  rankdir=LR;", '  node [shape=box, style=rounded];']
    gardes = set()
    for noeud in graphe.nodes:
        type_entite, _ = type_et_id(noeud)
        if type_entite not in types_retenus:
            continue
        gardes.add(noeud)
        nom = graphe.nodes[noeud].get("nom", noeud)
        etiquette = nom if len(nom) <= 38 else nom[:35] + "…"
        couleur = _COULEURS_NOEUDS.get(type_entite, "#555555")
        lignes.append(f'  "{noeud}" [label="{_echapper_dot(etiquette)}", color="{couleur}"];')
    for source, cible, donnees in graphe.edges(data=True):
        if source in gardes and cible in gardes:
            relation = _echapper_dot(str(donnees.get("relation", "")))
            lignes.append(f'  "{source}" -> "{cible}" [label="{relation}"];')
    lignes.append("}")
    return "\n".join(lignes)


def page_corpus_graphe() -> None:
    exiger_acces_admin()
    st.title("🕸️ Corpus et graphe de connaissances")
    if not _exiger_backend():
        return

    from src.models import charger_corpus_formations

    corpus = charger_corpus_formations()
    onglet_corpus, onglet_graphe = st.tabs(["Corpus structuré", "Graphe (ontologie)"])

    with onglet_corpus:
        for titre, donnees in (
            ("Mentions", corpus.mentions),
            ("Parcours", corpus.parcours),
            ("Prérequis d'admission", corpus.prerequis),
            ("Matières", corpus.matieres),
            ("Compétences", corpus.competences),
            ("Métiers", corpus.metiers),
        ):
            st.markdown(f"**{titre}** — {len(donnees)}")
            if donnees:
                st.dataframe([e.model_dump() for e in donnees], width="stretch")
            else:
                st.caption("Pas encore collecté (BACKLOG.md, DATA-1).")

    with onglet_graphe:
        st.caption(
            "Seules les relations **réellement présentes** dans le corpus sont tracées. "
            "`enseigne` / `developpe` / `prepareA` apparaîtront quand matières, "
            "compétences et débouchés auront été collectés (DATA-1)."
        )
        types = st.multiselect(
            "Types d'entités à afficher",
            sorted(_COULEURS_NOEUDS),
            default=["Parcours", "Mention", "Prerequis"],
        )
        if not types:
            st.info("Sélectionnez au moins un type d'entité.")
            return
        try:
            st.graphviz_chart(_graphe_dot(set(types)), width="stretch")
        except Exception as erreur:  # noqa: BLE001 — l'explorateur ne doit pas casser l'app
            # Message explicite : le silence précédent (« Graphe indisponible »)
            # ne permettait pas de distinguer un corpus vide d'un import cassé.
            st.error(f"Graphe non traçable : {type(erreur).__name__} — {erreur}", icon="⛔")


# --- Mesures ------------------------------------------------------------------


def _section_ml() -> None:
    resultats = charger_json_local("tests/eval_results_ml.json")
    if resultats is None:
        artefact_absent("L'évaluation ML", "cd backend && python -m tests.eval_ml")
        return

    # Le modèle **servi**, pas la baseline brute : les afficher l'un pour
    # l'autre reproduirait l'écart « évalué ≠ servi » que le §8 interdit.
    baseline = resultats.get("modele_de_production_calibre") or resultats.get(
        "baseline_regression_logistique", {}
    )
    st.caption(resultats.get("avertissement", ""))

    apport = resultats.get("apport_de_la_calibration")
    if apport:
        st.caption(
            f"Calibration isotonique : ECE {apport['ece_avant']:.3f} → "
            f"**{apport['ece_apres']:.3f}**. {apport['lecture']}"
        )

    st.markdown("**Modèle de production — régression logistique calibrée**")
    colonnes = st.columns(4)
    colonnes[0].metric("Exactitude", f"{baseline.get('exactitude', 0):.1%}", border=True)
    colonnes[1].metric("F1 macro", f"{baseline.get('f1_macro', 0):.3f}", border=True)
    colonnes[2].metric("MRR", f"{baseline.get('mrr', 0):.3f}", border=True)
    colonnes[3].metric("NDCG@3", f"{baseline.get('ndcg_3', 0):.3f}", border=True)

    calibration = baseline.get("calibration", {})
    st.markdown("**Calibration** — la confiance affichée est-elle juste ?")
    colonnes = st.columns(3)
    ece = calibration.get("ece")
    colonnes[0].metric("ECE", f"{ece:.3f}" if ece is not None else "—", border=True)
    colonnes[1].metric(
        "Score de Brier",
        f"{calibration.get('score_de_brier', 0):.3f}",
        border=True,
    )
    colonnes[2].metric(
        "PR-AUC macro", f"{baseline.get('pr_auc_macro') or 0:.3f}", border=True
    )
    # Le **sens** de l'écart est lu dans la mesure, jamais supposé : l'ECE est
    # une valeur absolue, et une alerte qui affirme « sur-confiance » sur un
    # modèle sous-confiant se trompe de diagnostic et de correctif.
    ecart = calibration.get("ecart_signe_confiance_moins_exactitude")
    if ece is not None and ece > 0.05 and ecart is not None:
        if ecart > 0:
            st.warning(
                f"**Sur-confiance de {abs(ecart):.0%}** : le modèle annonce en moyenne "
                "une certitude supérieure à sa réussite réelle. C'est le sens dangereux "
                "— un score d'adéquation surévalué présenté à un candidat.",
                icon="⚠️",
            )
        else:
            st.warning(
                f"**Sous-confiance de {abs(ecart):.0%}** : le modèle réussit plus "
                "souvent qu'il ne l'annonce. Moins dangereux qu'une sur-confiance, mais "
                "le score affiché ne correspond alors à aucune fréquence réelle.",
                icon="⚠️",
            )
    elif ece is not None:
        st.success(
            f"Calibration correcte (ECE {ece:.3f}) : le score affiché correspond, en "
            "moyenne, à la fréquence de réussite observée.",
            icon="✅",
        )

    production = resultats.get("chemin_de_production", {})
    if production:
        st.markdown("**Chemin réellement servi** (`analyser_profil`, §8)")
        st.caption(production.get("lecture", ""))
        classement = production.get("classement", {})
        colonnes = st.columns(4)
        colonnes[0].metric("Top-1", f"{classement.get('top_1', 0):.1%}", border=True)
        colonnes[1].metric("Top-3", f"{classement.get('top_3', 0):.1%}", border=True)
        colonnes[2].metric("MRR", f"{classement.get('mrr', 0):.3f}", border=True)
        colonnes[3].metric(
            "Profils jugés inexploitables",
            classement.get("profils_juges_inexploitables", 0),
            border=True,
        )

        stabilite = production.get("stabilite_des_recommandations", {})
        if stabilite.get("profils_compares"):
            st.markdown("**Stabilité des recommandations** (§7) — retrait d'un trait déclaré")
            top1 = stabilite.get("top_1_inchange", 0)
            presentee = stabilite.get("selection_presentee_inchangee", 0)
            top3_fixe = stabilite.get("top_3_fixe_inchange", 0)
            colonnes = st.columns(3)
            colonnes[0].metric("Parcours de tête inchangé", f"{top1:.0%}", border=True)
            colonnes[1].metric("Sélection présentée inchangée", f"{presentee:.0%}", border=True)
            colonnes[2].metric(
                "Top-3 fixe inchangé (référence)", f"{top3_fixe:.0%}", border=True
            )
            st.caption(
                "La mesure de référence est la **sélection réellement présentée** : "
                "seuls les parcours atteignant 20 % du score de tête sont proposés. "
                "La colonne « top-3 fixe » est conservée pour comparaison — elle "
                "mesurait la permanence des rangs 2 et 3, dont le score médian est de "
                "2,0 % et 1,1 %, c'est-à-dire du bruit."
            )
            if presentee < 0.85:
                st.warning(
                    f"La sélection présentée change pour **{1 - presentee:.0%}** des "
                    "profils au retrait d'un seul trait déclaré : à ne pas présenter "
                    "comme un classement arrêté.",
                    icon="⚠️",
                )

    with st.expander("Matrice de confusion (baseline)"):
        matrice = baseline.get("matrice_confusion", {})
        if matrice:
            st.dataframe(
                {
                    "vraie classe": matrice["labels"],
                    **{
                        predite: [ligne[i] for ligne in matrice["matrice"]]
                        for i, predite in enumerate(matrice["labels"])
                    },
                },
                width="stretch",
            )


def _section_rag() -> None:
    resultats = charger_json_local("tests/eval_results_rag_calibration.json")
    if resultats is None:
        artefact_absent(
            "La calibration RAG", "cd backend && python -m tests.calibrer_seuil_rag"
        )
        return

    meilleur = resultats.get("meilleur_compromis", {})
    st.markdown("**Balayage mesuré du seuil de pertinence et de k** (RAG-5)")
    colonnes = st.columns(4)
    colonnes[0].metric("Meilleur compromis — seuil", meilleur.get("seuil", "?"), border=True)
    colonnes[1].metric("Meilleur compromis — k", meilleur.get("k", "?"), border=True)
    colonnes[2].metric("Rappel", f"{meilleur.get('rappel', 0):.0%}", border=True)
    colonnes[3].metric(
        "Silence hors corpus",
        f"{meilleur.get('silence_correct_hors_corpus', 0):.0%}",
        border=True,
    )
    st.caption(
        "« Silence hors corpus » : part des questions sans réponse dans le corpus pour "
        "lesquelles le RAG ne renvoie **rien**. C'est un succès, pas un échec (§9) — "
        "répondre à partir de passages hors sujet est le mode de défaillance dangereux."
    )

    # La configuration livrée peut diverger du meilleur compromis brut, et cette
    # divergence doit être visible et justifiée plutôt que laisser croire à une
    # incohérence entre la mesure et le code.
    if _exiger_backend():
        from src.config import config

        st.markdown("**Configuration effectivement livrée**")
        gauche, droite = st.columns(2)
        gauche.metric("Seuil en vigueur", config.rag_seuil_pertinence, border=True)
        droite.metric("k en vigueur", config.rag_k, border=True)
        if config.rag_k != meilleur.get("k"):
            st.info(
                f"**k = {config.rag_k} et non {meilleur.get('k')}** : à seuil égal, rappel "
                "et silence sont identiques et la précision ne diffère que de deux points. "
                "Mais avec `rag_max_fragments_par_source = "
                f"{config.rag_max_fragments_par_source}`, "
                f"k = {meilleur.get('k')} plafonnerait à "
                f"{meilleur.get('k', 0) // max(config.rag_max_fragments_par_source, 1)} sources "
                "distinctes — trop peu pour les questions multi-sources exigées au §13.",
                icon="ℹ️",
            )

    st.dataframe(resultats.get("mesures", []), width="stretch")
    st.caption(resultats.get("limite", ""))


def _section_systeme() -> None:
    resultats = charger_json_local("tests/eval_results.json")
    if resultats is None:
        artefact_absent(
            "L'évaluation système (32 cas)", "cd backend && python -m tests.eval_system"
        )
        return

    reussis, total = resultats.get("reussis", 0), resultats.get("total", 0)
    colonnes = st.columns(3)
    colonnes[0].metric("Cas réussis", f"{reussis}/{total}", border=True)
    colonnes[1].metric(
        "Taux", f"{(reussis / total):.0%}" if total else "—", border=True
    )
    latence = resultats.get("latence_ms", {})
    colonnes[2].metric("Latence moyenne", f"{latence.get('moyenne', '?')} ms", border=True)

    st.warning(
        "**À remesurer avant la remise.** Ces chiffres datent d'avant la calibration "
        "du seuil RAG (RAG-5) : ils ne correspondent plus à la configuration livrée.",
        icon="⚠️",
    )

    par_categorie = resultats.get("taux_par_categorie", {})
    if par_categorie:
        st.markdown("**Résultat par catégorie du §13**")
        st.dataframe(
            [{"catégorie": c, "résultat": v} for c, v in par_categorie.items()],
            width="stretch",
        )
    with st.expander("Résultats détaillés"):
        st.json(resultats.get("resultats_detailles", []))


def page_mesures() -> None:
    exiger_acces_admin()
    st.title("📈 Mesures et évaluations")
    st.caption(
        "Le sujet exige des résultats **mesurés**, pas l'affirmation que le système "
        "fonctionne. Chaque chiffre vient d'un artefact du dépôt, reproductible par "
        "la commande indiquée."
    )
    onglet_ml, onglet_rag, onglet_systeme = st.tabs(
        ["Machine Learning", "Recherche documentaire", "Système de bout en bout"]
    )
    with onglet_ml:
        _section_ml()
    with onglet_rag:
        _section_rag()
    with onglet_systeme:
        _section_systeme()
