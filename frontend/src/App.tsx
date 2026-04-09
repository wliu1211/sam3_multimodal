import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import PromptInput from "./components/PromptInput";
import ResultCanvas from "./components/ResultCanvas";
import ColorLegend from "./components/ColorLegend";
import LoadingSpinner from "./components/LoadingSpinner";
import QueryHistory from "./components/QueryHistory";
import { useQuerySubmit } from "./hooks/useQuerySubmit";
import { useQueryHistory } from "./hooks/useQueryHistory";

export default function App() {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const { submit, result, loading, error, reset } = useQuerySubmit();
  const { history, refresh: refreshHistory } = useQueryHistory();

  const handleImageSelect = (file: File) => {
    setImage(file);
    setImagePreviewUrl(URL.createObjectURL(file));
    reset();
  };

  const handleSubmit = async (prompt: string) => {
    if (!image) return;
    await submit(image, prompt);
    refreshHistory();
  };

  const handleReset = () => {
    setImage(null);
    setImagePreviewUrl(null);
    reset();
  };

  return (
    <div className="app-shell">
      {/* Top nav */}
      <header className="topbar">
        <div className="topbar-left">
          <span className="logo">SVQ</span>
          <span className="logo-sub">Semantic Visual Query</span>
        </div>
        <button
          className="history-toggle"
          onClick={() => setSidebarOpen((v) => !v)}
          aria-label="Toggle query history"
        >
          <HistoryIcon />
          <span>History</span>
        </button>
      </header>

      <div className="main-layout">
        {/* Left panel — input controls */}
        <aside className="input-panel">
          <section className="panel-section">
            <label className="section-label">1 — Upload image</label>
            <ImageUploader
              onImageSelect={handleImageSelect}
              previewUrl={imagePreviewUrl}
              onClear={handleReset}
            />
          </section>

          <section className="panel-section">
            <label className="section-label">2 — Describe what to find</label>
            <PromptInput
              onSubmit={handleSubmit}
              disabled={!image || loading}
              loading={loading}
            />
          </section>

          {result && (
            <section className="panel-section">
              <label className="section-label">Matched groups</label>
              <ColorLegend groups={result.groups}/>
            </section>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <span>⚠ {error}</span>
            </div>
          )}
        </aside>

        {/* Center — result viewer */}
        <main className="canvas-panel">
          {loading && (
            <div className="canvas-overlay">
              <LoadingSpinner />
            </div>
          )}

          {!loading && !result && !imagePreviewUrl && (
            <div className="empty-state">
              <div className="empty-icon">
                <SearchIcon />
              </div>
              <p className="empty-title">Upload an image and describe what to find</p>
              <p className="empty-sub">
                Try: <em>"find cars that are blue and a Honda"</em>
              </p>
            </div>
          )}

          {(imagePreviewUrl || result) && !loading && (
            <ResultCanvas
              imageUrl={imagePreviewUrl!}
              groups={result?.groups ?? []}
              isAnnotated={!!result}
              sceneGraph={result?.sceneGraph}
              relationalHints={result?.parsedQuery.relationalHints}
            />
          )}
        </main>

        {/* Right sidebar — query history */}
        <QueryHistory
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          history={history}
        />
      </div>
    </div>
  );
}

function HistoryIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M8 4.5V8l2.5 2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <circle cx="17" cy="17" r="10" stroke="currentColor" strokeWidth="2" />
      <path d="M24.5 24.5L33 33" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      <path d="M12 17h10M17 12v10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}