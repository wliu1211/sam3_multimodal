const STAGES = [
    { label: "Parsing prompt",        detail: "LLM extracting base class + attributes" },
    { label: "Detecting objects",     detail: "SAM 3 scanning image for matches" },
    { label: "Verifying attributes",  detail: "CLIP scoring color, type, make" },
    { label: "Rendering masks",       detail: "Drawing overlays onto image" },
  ];
  
  interface Props {
    stage?: number; // 0-indexed current stage, optional
  }
  
  export default function LoadingSpinner({ stage }: Props) {
    const activeStage = stage ?? -1; // -1 = indeterminate (cycle animation)
  
    return (
      <div className="spinner-wrapper" role="status" aria-label="Running inference">
        {/* Animated ring */}
        <div className="spinner-ring" aria-hidden="true">
          <svg viewBox="0 0 44 44" className="spinner-svg">
            <circle
              cx="22" cy="22" r="18"
              fill="none"
              stroke="var(--border)"
              strokeWidth="3"
            />
            <circle
              cx="22" cy="22" r="18"
              fill="none"
              stroke="var(--accent)"
              strokeWidth="3"
              strokeLinecap="round"
              strokeDasharray="30 84"
              className="spinner-arc"
            />
          </svg>
        </div>
  
        <p className="spinner-title">Running inference</p>
  
        {/* Pipeline stages */}
        <ol className="spinner-stages">
          {STAGES.map((s, i) => {
            const isDone    = activeStage > i;
            const isActive  = activeStage === i || activeStage === -1;
            return (
              <li
                key={s.label}
                className={`spinner-stage ${isDone ? "spinner-stage--done" : ""} ${isActive ? "spinner-stage--active" : ""}`}
              >
                <span className="stage-dot" aria-hidden="true">
                  {isDone ? "✓" : isActive ? "›" : "·"}
                </span>
                <span className="stage-text">
                  <span className="stage-label">{s.label}</span>
                  {isActive && <span className="stage-detail">{s.detail}</span>}
                </span>
              </li>
            );
          })}
        </ol>
      </div>
    );
  }