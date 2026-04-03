import { useRef, useState } from "react";
import type { DragEvent, ChangeEvent } from "react";
interface Props {
  onImageSelect: (file: File) => void;
  previewUrl: string | null;
  onClear: () => void;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE_MB = 20;

export default function ImageUploader({ onImageSelect, previewUrl, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  const validate = (file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type))
      return "Only JPEG, PNG, or WebP images are supported.";
    if (file.size > MAX_SIZE_MB * 1024 * 1024)
      return `Image must be under ${MAX_SIZE_MB}MB.`;
    return null;
  };

  const handleFile = (file: File) => {
    const err = validate(file);
    if (err) { setFileError(err); return; }
    setFileError(null);
    onImageSelect(file);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  };

  const onDragLeave = () => setDragging(false);

  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // reset input so same file can be re-selected
    e.target.value = "";
  };

  if (previewUrl) {
    return (
      <div className="uploader-preview">
        <img src={previewUrl} alt="Uploaded preview" className="preview-img" />
        <button className="preview-clear" onClick={onClear} aria-label="Remove image">
          <CloseIcon />
          <span>Remove</span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={`uploader-dropzone ${dragging ? "uploader-dropzone--active" : ""}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload image"
      onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="uploader-input-hidden"
        onChange={onInputChange}
        aria-hidden="true"
        tabIndex={-1}
      />

      <div className="uploader-icon">
        <UploadIcon />
      </div>
      <p className="uploader-title">Drop an image here</p>
      <p className="uploader-sub">or click to browse — JPEG, PNG, WebP up to {MAX_SIZE_MB}MB</p>

      {fileError && (
        <p className="uploader-error" role="alert">{fileError}</p>
      )}
    </div>
  );
}

function UploadIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <rect x="1" y="1" width="26" height="26" rx="6" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3" />
      <path d="M14 19V9M9 14l5-5 5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}