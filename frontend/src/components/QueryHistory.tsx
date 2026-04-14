import { useEffect, useRef } from "react";
import type { QueryLogEntry } from "../types/history";
interface Props {
    open: boolean;
    onClose: () => void;
    history: QueryLogEntry[];
}
export default function QueryHistory({ open, onClose, history }: Props) {
    const panelRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (!open)
            return;
        const handler = (e: KeyboardEvent) => { if (e.key === "Escape")
            onClose(); };
        document.addEventListener("keydown", handler);
        return () => document.removeEventListener("keydown", handler);
    }, [open, onClose]);
    useEffect(() => {
        if (open)
            panelRef.current?.focus();
    }, [open]);
    return (<>
      
      {open && (<div className="sidebar-backdrop" onClick={onClose} aria-hidden="true"/>)}

      
      <aside ref={panelRef} className={`history-sidebar ${open ? "history-sidebar--open" : ""}`} aria-label="Query history" tabIndex={-1} role="complementary">
        <div className="history-header">
          <h2 className="history-title">Query history</h2>
          <button className="history-close" onClick={onClose} aria-label="Close history">
            <CloseIcon />
          </button>
        </div>

        <div className="history-body">
          {history.length === 0 ? (<div className="history-empty">
              <p>No queries yet.</p>
              <p className="history-empty-sub">Your past searches will appear here.</p>
            </div>) : (<ol className="history-list" reversed>
              {[...history].reverse().map((entry) => (<li key={entry.id} className="history-item">
                  
                  {entry.resultImageUrl && (<img src={entry.resultImageUrl} alt="" className="history-thumb" loading="lazy"/>)}

                  <div className="history-item-body">
                    <p className="history-prompt" title={entry.prompt}>
                      {entry.prompt}
                    </p>
                    <div className="history-meta">
                      <span className="history-time">
                        {formatRelativeTime(entry.createdAt)}
                      </span>
                      {entry.matchCount !== undefined && (<span className="history-matches">
                          {entry.matchCount} match{entry.matchCount !== 1 ? "es" : ""}
                        </span>)}
                    </div>
                  </div>
                </li>))}
            </ol>)}
        </div>
      </aside>
    </>);
}
function formatRelativeTime(isoString: string): string {
    const diff = Date.now() - new Date(isoString).getTime();
    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    if (mins < 1)
        return "just now";
    if (mins < 60)
        return `${mins}m ago`;
    if (hours < 24)
        return `${hours}h ago`;
    return `${days}d ago`;
}
function CloseIcon() {
    return (<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>);
}
