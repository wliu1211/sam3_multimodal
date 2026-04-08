import type { MaskGroup } from "../types/mask";

interface Props {
  groups: MaskGroup[];
}

export default function ColorLegend({ groups }: Props) {
  if (groups.length === 0) return null;

  return (
    <div className="color-legend" role="list" aria-label="Matched groups legend">
      {groups.map((group) => (
        <div
          key={group.label}
          className="legend-row"
          role="listitem"
        >
          <div
            className="legend-swatch"
            style={{ background: group.color }}
            aria-hidden="true"
          />

          <div className="legend-info">
            <span className="legend-label">{group.label}</span>
            {group.attributes.length > 0 && (
              <span className="legend-attrs">
                {group.attributes.join(" · ")}
              </span>
            )}
          </div>

          <span
            className="legend-count"
            style={{ background: group.color + "22", color: group.color, borderColor: group.color + "44" }}
            aria-label={`${group.matchCount} matches`}
          >
            {group.matchCount}
          </span>

          {group.avgConfidence > 0 && (
            <div className="legend-confidence" title={`Avg confidence: ${Math.round(group.avgConfidence * 100)}%`}>
              <div
                className="legend-confidence-fill"
                style={{
                  width: `${group.avgConfidence * 100}%`,
                  background: group.color,
                }}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}