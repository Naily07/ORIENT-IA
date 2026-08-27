import { CorpusGrapheView } from "@/components/admin/CorpusGrapheView";
import { ErreurChargement } from "@/components/admin/ErreurChargement";
import { getCorpus, getGraphe } from "@/lib/admin-api";

async function chargerDonnees() {
  try {
    const [corpus, graphe] = await Promise.all([getCorpus(), getGraphe()]);
    return { ok: true as const, corpus, graphe };
  } catch (erreur) {
    return {
      ok: false as const,
      message: erreur instanceof Error ? erreur.message : "Erreur inconnue.",
    };
  }
}

export default async function PageCorpus() {
  const resultat = await chargerDonnees();
  if (!resultat.ok) {
    return <ErreurChargement message={`Corpus indisponible : ${resultat.message}`} />;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Corpus et graphe de connaissances</h1>
      <CorpusGrapheView corpus={resultat.corpus} graphe={resultat.graphe} />
    </div>
  );
}
