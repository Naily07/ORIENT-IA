import { AlertTriangle } from "lucide-react";

export function ErreurChargement({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
      <AlertTriangle className="size-4 shrink-0" aria-hidden="true" />
      {message}
    </div>
  );
}
