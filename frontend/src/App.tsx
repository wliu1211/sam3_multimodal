import { useState } from "react";

import ImageUploader from "./components/ImageUploader";
import PromptInput from "./components/PromptInput";
import ResultCanvas from "./components/ResultCanvas";
import ColorLegend from "./components/ColorLegend";
import LoadingSpinner from "./components/LoadingSpinner";
import QueryHistory from "./components/QueryHistory";
import { useQuerySubmit } from "./hooks/useQuerySubmit";
import { useQueryHistory } from "./hooks/useQueryHistory";

import ActivityList from "./components/ActivityList";
import { analyzeImage, maskGroup } from "./api/endpoints";
import type { DiscoveryResponse, ActivityGroup } from "./types/query";

export default function App() {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mode, setMode] = useState<"search" | "discover">("search");

  const [discovery, setDiscovery] = useState<DiscoveryResponse | null>(null);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [maskedImageUrl, setMaskedImageUrl]   = useState<string | null>(null);
  const [discoverLoading, setDiscoverLoading] = useState(false);
  const [maskLoading, setMaskLoading]         = useState(false);
  const [discoverError, setDiscoverError]     = useState<string | null>(null);

  const { submit, result, loading, error, reset } = useQuerySubmit();
  const { history, refresh: refreshHistory } = useQueryHistory();

  const handleImageSelect = (file: File) => {
    setImage(file);
    setImagePreviewUrl(URL.createObjectURL(file));
    reset();
    setDiscovery(null);
    setMaskedImageUrl(null);
    setSelectedGroupId(null);
    setDiscoverError(null);
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
    setDiscovery(null);
    setMaskedImageUrl(null);
    setSelectedGroupId(null);
    setDiscoverError(null);
  };

  const handleDescribeImage = async () => {
    if (!image) return;
    setDiscovery(null);
    setMaskedImageUrl(null);
    setSelectedGroupId(null);
    setDiscoverError(null);
    setDiscoverLoading(true);

    try {
      const response = await analyzeImage(image);
      setDiscovery(response);
    } catch (err) {
      setDiscoverError(err instanceof Error ? err.message : "Scene analysis failed.");
    } finally {
      setDiscoverLoading(false);
    }
  };

  const handleActivitySelect = async (group: ActivityGroup) => {
    if (!discovery) return;
    setSelectedGroupId(group.id);
    setMaskLoading(true);

    try {
      const result = await maskGroup(discovery.id, group.node_ids, group.color);
      setMaskedImageUrl(result.annotatedImageUrl);
    } catch (err) {
      setDiscoverError(err instanceof Error ? err.message : "Masking failed.");
    } finally {
      setMaskLoading(false);
    }
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

          {/* Mode toggle */}
          <div className="mode-toggle-group">
            <button
              className={`mode-toggle-btn${mode === "search" ? " mode-toggle-btn--active" : ""}`}
              onClick={() => setMode("search")}
            >
              Search
            </button>
            <button
              className={`mode-toggle-btn${mode === "discover" ? " mode-toggle-btn--active" : ""}`}
              onClick={() => setMode("discover")}
            >
              Discover
            </button>
          </div>

          {mode === "search" && (
            <section className="panel-section">
              <label className="section-label">2 — Describe what to find</label>
              <PromptInput
                onSubmit={handleSubmit}
                disabled={!image || loading}
                loading={loading}
              />
            </section>
          )}

          {mode === "discover" && (
            <section className="panel-section">
              <button
                className="describe-image-btn"
                disabled={!image || loading}
                onClick={handleDescribeImage}
              >
                <DescribeImageIcon />
                Describe Image
              </button>

              {discoverLoading && (
                <div className="discover-loading">
                  <LoadingSpinner />
                  <p className="loading-label">Analyzing scene…</p>
                </div>
              )}

              {!discoverLoading && !discovery && (
                <p className="activities-empty">
                  Activities will appear here after describing the image.
                </p>
              )}

              {!discoverLoading && discovery && (
                <>
                  <p className="activities-count">
                    {discovery.activities.length} activit{discovery.activities.length === 1 ? "y" : "ies"} detected
                  </p>
                  <ActivityList
                    activities={discovery.activities}
                    selectedId={selectedGroupId}
                    maskLoading={maskLoading}
                    onSelect={handleActivitySelect}
                  />
                </>
              )}

              {discoverError && (
                <p className="activities-error">⚠ {discoverError}</p>
              )}
            </section>
          )}

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
              imageUrl={
                mode === "discover"
                  ? (maskedImageUrl ?? imagePreviewUrl!)
                  : result
                    ? result.annotatedImageUrl
                    : imagePreviewUrl!
              }
              groups={mode === "discover" ? [] : (result?.groups ?? [])}
              isAnnotated={mode === "discover" ? !!maskedImageUrl : !!result}
              sceneGraph={mode === "discover" ? undefined : result?.sceneGraph}
              relationalHints={mode === "discover" ? undefined : result?.parsedQuery.relationalHints}
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

function DescribeImageIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <rect x="1" y="1" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="4.5" cy="4.5" r="1" fill="currentColor" />
      <path d="M1 9.5l3-3 2.5 2.5L9 6l4 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}