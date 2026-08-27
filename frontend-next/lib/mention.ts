/**
 * Repli local de la mention obligatoire (§16 du sujet, SEC-5) si l'API est
 * injoignable — copie de `src.config.MENTION_OBLIGATOIRE`, même mécanisme de
 * repli que `frontend/noyau.py`. Source unique réelle : `GET /health`.
 */
export const MENTION_OBLIGATOIRE_REPLI =
  "ORIENT'IA constitue un outil d'aide à l'orientation. Ses recommandations " +
  "ne remplacent ni l'avis d'un conseiller pédagogique ni une décision " +
  "officielle d'admission.";
