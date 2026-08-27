import dagre from "dagre";

/** Calcule un layout hiérarchique (gauche→droite) pour le graphe de
 * connaissances — reproduit `rankdir=LR` du DOT généré par
 * `back_office._graphe_dot`, qui n'a pas d'équivalent direct en React. */
export const LARGEUR_NOEUD = 172;
export const HAUTEUR_NOEUD = 40;

export function calculerLayout(
  noeuds: { id: string }[],
  relations: { source: string; cible: string }[],
): Record<string, { x: number; y: number }> {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 24, ranksep: 80 });
  g.setDefaultEdgeLabel(() => ({}));

  for (const noeud of noeuds) {
    g.setNode(noeud.id, { width: LARGEUR_NOEUD, height: HAUTEUR_NOEUD });
  }
  for (const relation of relations) {
    g.setEdge(relation.source, relation.cible);
  }

  dagre.layout(g);

  const positions: Record<string, { x: number; y: number }> = {};
  for (const noeud of noeuds) {
    const position = g.node(noeud.id);
    // dagre positionne le centre du nœud ; React Flow attend le coin
    // supérieur gauche.
    positions[noeud.id] = { x: position.x - LARGEUR_NOEUD / 2, y: position.y - HAUTEUR_NOEUD / 2 };
  }
  return positions;
}
