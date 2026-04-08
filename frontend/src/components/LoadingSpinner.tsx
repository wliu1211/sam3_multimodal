export default function LoadingSpinner() {
  return (
    <div className="spinner-wrapper" role="status" aria-label="Running inference">
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
    </div>
  );
}