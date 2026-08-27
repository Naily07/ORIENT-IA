"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export function LimiteSelector({ limite }: { limite: number }) {
  const router = useRouter();
  const [valeur, setValeur] = useState(limite);

  return (
    <label className="flex items-center gap-2 text-xs text-neutral-500">
      Nombre de traces
      <input
        type="range"
        min={5}
        max={200}
        value={valeur}
        onChange={(evenement) => setValeur(Number(evenement.target.value))}
        onMouseUp={() => router.push(`/admin/observabilite?limite=${valeur}`)}
        onTouchEnd={() => router.push(`/admin/observabilite?limite=${valeur}`)}
        className="w-32"
      />
      <span className="w-8 tabular-nums">{valeur}</span>
    </label>
  );
}
