"""Interface ORIENT'IA — point d'entrée et navigation (FE-1 à FE-4, SEC-5).

Deux espaces distincts, pour deux publics qui n'ont pas les mêmes questions :

- **Espace candidat** — une page, en langage courant, qui répond à « quelle
  formation pour moi ? » et dit toujours sur quoi elle s'appuie.
- **Administration** — cinq vues pour l'équipe et le jury, qui répondent à
  « ce système est-il fiable, et comment le sait-on ? », chacune adossée à un
  artefact reproductible du dépôt.

Les séparer n'est pas cosmétique : le vocabulaire interne (`escalade_conseiller`,
ECE, NDCG, traces JSONL) est indispensable au jury et hors de propos devant un
candidat, tandis que les avertissements réglementaires du §16 doivent
accompagner le candidat en permanence.

**Toutes les pages sont déclarées dans une seule `st.navigation`**, groupées en
sections. Une première version conditionnait la liste des pages à un sélecteur
d'espace : les liens profonds cessaient alors de fonctionner — ouvrir
`/page_mesures`, ou simplement recharger la page en cours d'administration,
renvoyait au front-office avec « The page that you have requested does not seem
to exist » dans la console. Le contrôle d'accès vit donc dans les pages
elles-mêmes (`noyau.exiger_acces_admin`), pas dans le routage.

Lancement :

    ./run.sh                        # API + interface
    streamlit run frontend/app.py   # interface seule
"""

from __future__ import annotations

import streamlit as st

import back_office
import front_office
from noyau import API, bandeau_degradations, puce_etat_api

st.set_page_config(
    page_title="ORIENT'IA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Espace candidat": [
        st.Page(front_office.page, title="Mon orientation", icon="🎓", default=True),
    ],
    "Administration": [
        st.Page(back_office.page_tableau_de_bord, title="Tableau de bord", icon="🛠️"),
        st.Page(back_office.page_mesures, title="Mesures", icon="📈"),
        st.Page(back_office.page_observabilite, title="Observabilité", icon="📊"),
        st.Page(back_office.page_qualite_donnees, title="Qualité des données", icon="🔎"),
        st.Page(back_office.page_corpus_graphe, title="Corpus et graphe", icon="🕸️"),
    ],
}

with st.sidebar:
    st.title("🎓 ORIENT'IA")
    st.caption("Assistant d'orientation pédagogique — ISPM")
    puce_etat_api()
    st.caption(f"API : `{API}`")
    st.divider()

page = st.navigation(PAGES, position="sidebar")

# Les dégradations qui changent la lecture de l'écran sont annoncées avant
# toute page, dans les deux espaces.
bandeau_degradations()

page.run()
