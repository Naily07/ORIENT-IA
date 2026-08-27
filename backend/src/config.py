"""Configuration centralisée du projet.

Un seul endroit pour les réglages qui changent entre le développement, les
tests et la démo (modèle LLM, seuils, chemins). Évite les `os.environ.get`
dispersés dans chaque module et rend les paramètres d'évaluation (seuil RAG,
limite d'itérations de l'agent) visibles et justifiables devant le jury.

Repris de l'infrastructure d'un hackathon ISPM précédent (mAIntenance &
Assistance) : le mécanisme est générique, seules les valeurs métier
(catégories, seuils calibrés sur un autre corpus) restent à ajuster au fil de
l'avancement d'ORIENT'IA.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py vit dans backend/src/ : deux .parent remontent à backend/, où
# vivent data/, logs/ et chroma_db/.
RACINE = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    # Chemin absolu, pas ".env" relatif : un env_file relatif se résout par
    # rapport au cwd du process au démarrage, pas à l'emplacement de ce
    # fichier. `src.api` doit être lancé depuis backend/ pour que `src` soit
    # importable ; un ".env" relatif chercherait alors backend/.env, qui
    # n'existe pas — il reste à la racine du dépôt, à côté de frontend/.
    model_config = SettingsConfigDict(env_file=str(RACINE.parent / ".env"), extra="ignore")

    # --- LLM (Google AI Studio) ---
    gemini_api_key: str = ""
    # Flash-Lite : modèle Free Tier au débit le plus élevé de la gamme Gemini.
    # Le pipeline fait plusieurs appels LLM par requête (profil, RAG,
    # explication), donc le débit prime sur la profondeur de raisonnement
    # pour la majorité de ces appels.
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_max_output_tokens: int = 2048
    llm_temperature: float = 0.0  # déterminisme : décisions reproductibles
    # Free Tier : ~15 requêtes/minute constatées sur ce modèle. Le pipeline
    # émettant plusieurs appels par requête, la reprise sur quota n'est pas
    # optionnelle.
    llm_max_tentatives: int = 4
    llm_attente_quota: float = 6.0  # secondes, si l'API n'indique pas de délai
    # Timeout d'un appel HTTP au modèle. Sans borne explicite, une API qui ne
    # répond pas fige la requête FastAPI et le frontend attend indéfiniment.
    llm_timeout_s: float = 30.0
    # Lissage proactif : espacer les appels coûte moins cher que d'encaisser
    # une 429, dont le délai de reprise imposé par l'API dépasse la minute.
    # 0 désactive le lissage.
    llm_requetes_par_minute: int = 14  # marge sous la limite Free Tier de 15

    # --- Sortie structurée ---
    # Nombre de tentatives de génération conforme au schéma avant d'abandonner.
    llm_max_essais_validation: int = 2

    # --- RAG ---
    # ATTENTION : ChromaDB utilise L2 au carré par défaut, pas le cosinus. La
    # collection est créée explicitement en espace cosinus (voir rag.py) —
    # sans cela, ce seuil s'appliquerait à une échelle deux fois plus grande.
    #
    # **Valeurs calibrées sur le corpus ISPM** (RAG-5), pas héritées : voir
    # `backend/tests/calibrer_seuil_rag.py` et
    # `backend/tests/eval_results_rag_calibration.json`. Les valeurs
    # précédentes (0.75 / k=8) venaient d'un corpus de support informatique et
    # se sont révélées franchement mauvaises ici : rappel parfait, mais
    # **silence nul** sur les questions hors corpus — le RAG renvoyait toujours
    # des passages, y compris quand le corpus n'a rien à dire, ce qui invite le
    # modèle à broder (§16) — et une précision de 0,17, soit 83 % de passages
    # hors sujet dans le contexte.
    #
    # Mesuré sur 12 questions à source connue et 4 questions hors corpus :
    #
    #   seuil  k   rappel  précision  silence hors corpus
    #    0.56  5     0.75       0.48                 1.00   <- retenu
    #    0.60  5     1.00       0.47                 0.25
    #    0.75  8     1.00       0.17                 0.00   <- ancienne valeur
    #
    # Le compromis retenu privilégie le silence : ne rien trouver est un
    # comportement attendu du sujet (§9, « reconnaître les situations dans
    # lesquelles les informations disponibles ne permettent pas de conclure »),
    # là où répondre à partir de passages hors sujet est le mode d'échec
    # dangereux. Le rappel perdu est en partie rattrapé par les outils
    # structurés de l'agent, qui n'ont pas besoin du RAG pour répondre.
    #
    # k=5 plutôt que k=3 (rappel et silence identiques, précision à 0,02 près) :
    # avec `rag_max_fragments_par_source=2`, k=3 plafonne à 2 sources distinctes,
    # trop peu pour les questions multi-sources exigées au §13.
    rag_seuil_pertinence: float = 0.56
    rag_k: int = 5
    rag_taille_chunk: int = 220  # en mots
    rag_chevauchement: int = 40
    # Empêche un article long de monopoliser le top-k avec ses propres
    # fragments, au détriment d'une seconde source pertinente.
    rag_max_fragments_par_source: int = 2
    rag_collection: str = "corpus-pedagogique"

    # --- Agent ---
    agent_max_iterations: int = 5  # contrôle du nombre d'actions de l'agent

    # --- Orchestrateur ---
    # Budget de temps total d'une requête. Au-delà, les étapes optionnelles
    # du pipeline sont sautées et la décision est construite avec ce qui a
    # déjà été obtenu, plutôt que de laisser l'utilisateur attendre.
    orchestrateur_budget_s: float = 120.0
    # En dessous de ce seuil, la recommandation est trop incertaine pour être
    # présentée sans renvoyer vers un conseiller pédagogique ou l'administration.
    orchestrateur_seuil_confiance: float = 0.5

    # --- Chemins ---
    dossier_data: Path = RACINE / "data"
    dossier_logs: Path = RACINE / "logs"
    dossier_chroma: Path = RACINE / "chroma_db"

    @property
    def fichier_traces(self) -> Path:
        return self.dossier_logs / "traces.jsonl"

    @property
    def fichier_tool_calls(self) -> Path:
        return self.dossier_logs / "tool_calls.jsonl"

    @property
    def fichier_llm_calls(self) -> Path:
        return self.dossier_logs / "llm_calls.jsonl"


config = Config()

# Mention obligatoire dans l'interface (§16 du sujet, SEC-5) : reprise mot
# pour mot. Une constante plutôt qu'une valeur de `Config` — ce n'est pas un
# réglage ajustable par environnement, c'est un texte imposé par le sujet.
# Exposée via `GET /health` en attendant qu'un frontend (FE-1) l'affiche
# réellement à l'écran ; ce module reste la source unique du texte exact.
MENTION_OBLIGATOIRE = (
    "ORIENT'IA constitue un outil d'aide à l'orientation. Ses recommandations "
    "ne remplacent ni l'avis d'un conseiller pédagogique ni une décision "
    "officielle d'admission."
)
