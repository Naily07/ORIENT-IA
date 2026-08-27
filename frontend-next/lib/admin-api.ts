/**
 * Accesseurs typés vers les endpoints `/admin/*` (`backend/src/admin_api.py`)
 * et `/observabilite/traces` (admin-only côté usage, pas côté protection —
 * voir le docstring de `admin_api.py`).
 */
import { apiGet } from "@/lib/api-client";
import type {
  CorpusFormations,
  GrapheReponse,
  Intervalle,
  MesuresReponse,
  QualiteDonneesReponse,
  TableauDeBordReponse,
  TendancesReponse,
  Trace,
} from "@/lib/types";

export function getTableauDeBord(): Promise<TableauDeBordReponse> {
  return apiGet<TableauDeBordReponse>("/admin/tableau-de-bord");
}

export function getTendances(intervalle: Intervalle, limite?: number): Promise<TendancesReponse> {
  return apiGet<TendancesReponse>("/admin/observabilite/tendances", { intervalle, limite });
}

export function getTraces(limite?: number): Promise<Trace[]> {
  return apiGet<Trace[]>("/observabilite/traces", { limite });
}

export function getQualiteDonnees(): Promise<QualiteDonneesReponse> {
  return apiGet<QualiteDonneesReponse>("/admin/qualite-donnees");
}

export function getCorpus(): Promise<CorpusFormations> {
  return apiGet<CorpusFormations>("/admin/corpus");
}

export function getGraphe(types?: string[]): Promise<GrapheReponse> {
  return apiGet<GrapheReponse>("/admin/graphe", types ? { types } : undefined);
}

export function getMesures(): Promise<MesuresReponse> {
  return apiGet<MesuresReponse>("/admin/mesures");
}
