// CHANGED: added SceneGraph to the type import
import type { MaskGroup, SceneGraph } from "../types/mask";

// CHANGED: added optional sceneGraph prop
interface Props {
  groups: MaskGroup[];
  sceneGraph?: SceneGraph;
}

// CHANGED: destructure sceneGraph from props
export default function ColorLegend({ groups, sceneGraph }: Props) {
  if (groups.length === 0) return null;

  // Build a lookup of node id → label for rendering edge subject/object names
  const nodeLabel = (id: string): string => {
    const node = sceneGraph?.nodes.find((n) => n.id === id);
    return node ? node.label : id;
  };

  // Only show verified edges in the legend
  const verifiedEdges = sceneGraph?.edges.filter((e) => e.verified) ?? [];

  return (
    <div className="color-legend" role="list" aria-label="Matched groups legend">

      {/* ── Existing: matched groups ── */}
      {groups.map((group) => (
        <div
          key={group.label}
          className="legend-row"
          role="listitem"
        >
          {/* Color swatch */}
          <div
            className="legend-swatch"
            style={{ background: group.color }}
            aria-hidden="true"
          />

          {/* Label + confidence */}
          <div className="legend-info">
            <span className="legend-label">{group.label}</span>
            {group.attributes.length > 0 && (
              <span className="legend-attrs">
                {group.attributes.join(" · ")}
              </span>
            )}
          </div>

          {/* Match count pill */}
          <span
            className="legend-count"
            style={{ background: group.color + "22", color: group.color, borderColor: group.color + "44" }}
            aria-label={`${group.matchCount} matches`}
          >
            {group.matchCount}
          </span>

          {/* Average confidence bar */}
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

      {/* ── NEW: verified relationships section ── */}
      {verifiedEdges.length > 0 && (
        <>
          <div className="legend-section-divider" aria-hidden="true" />
          <p className="legend-section-label">Relationships</p>

          {verifiedEdges.map((edge, i) => (
            <div
              key={`${edge.subjectId}-${edge.predicate}-${edge.objectId}-${i}`}
              className="legend-row legend-row--edge"
              role="listitem"
              title={
                edge.downgraded && edge.originalPredicate
                  ? `Downgraded from "${edge.originalPredicate}"`
                  : undefined
              }
            >
              {/* Edge direction indicator */}
              <div className="legend-edge-arrow" aria-hidden="true">→</div>

              {/* Subject · predicate · Object */}
              <div className="legend-info">
                <span className="legend-label">
                  <span className="legend-edge-subject">{nodeLabel(edge.subjectId)}</span>
                  <span className="legend-edge-pred">
                    {edge.downgraded ? "~" : ""}{edge.predicate.replace(/_/g, " ")}
                  </span>
                  <span className="legend-edge-object">{nodeLabel(edge.objectId)}</span>
                </span>
              </div>

              {/* Confidence pill */}
              <span
                className="legend-count"
                style={{
                  background: "var(--bg-raised)",
                  color: "var(--text-secondary)",
                  borderColor: "var(--border)",
                }}
                aria-label={`${Math.round(edge.confidence * 100)}% confidence`}
              >
                {Math.round(edge.confidence * 100)}%
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}