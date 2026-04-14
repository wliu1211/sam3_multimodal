import type { ActivityGroup } from "../types/query";
interface Props {
    activities: ActivityGroup[];
    selectedId: string | null;
    maskLoading: boolean;
    onSelect: (group: ActivityGroup) => void;
}
export default function ActivityList({ activities, selectedId, maskLoading, onSelect, }: Props) {
    if (activities.length === 0)
        return null;
    return (<div className="activity-list" role="list" aria-label="Detected activities">
      {activities.map((group) => {
            const isSelected = group.id === selectedId;
            const isLoading = isSelected && maskLoading;
            return (<button key={group.id} className={`activity-item ${isSelected ? "activity-item--active" : ""}`} onClick={() => onSelect(group)} disabled={maskLoading} role="listitem" aria-pressed={isSelected} aria-label={`${group.label}, confidence ${Math.round(group.confidence * 100)}%`}>
            
            <div className="activity-swatch" style={{ background: group.color }} aria-hidden="true"/>

            
            <div className="activity-info">
              <span className="activity-label">{group.label}</span>
              <div className="activity-confidence-bar" title={`Confidence: ${Math.round(group.confidence * 100)}%`}>
                <div className="activity-confidence-fill" style={{
                    width: `${group.confidence * 100}%`,
                    background: group.color,
                }}/>
              </div>
            </div>

            
            <div className="activity-right">
              {isLoading && (<span className="activity-spinner" aria-label="Loading masks">
                  <SpinnerIcon />
                </span>)}
            </div>
          </button>);
        })}
    </div>);
}
function SpinnerIcon() {
    return (<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" style={{ animation: "spin 0.8s linear infinite" }}>
      <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.5" strokeDasharray="22" strokeDashoffset="8" strokeLinecap="round"/>
    </svg>);
}
