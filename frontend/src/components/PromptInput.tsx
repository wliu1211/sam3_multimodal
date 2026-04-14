import { useState } from "react";
import type { KeyboardEvent } from "react";
interface Props {
    onSubmit: (prompt: string) => void;
    disabled: boolean;
    loading: boolean;
}
const EXAMPLES = [
    "find cars that are blue and a Honda",
    "find people wearing red jackets",
    "find large brown dogs",
    "find stop signs",
];
export default function PromptInput({ onSubmit, disabled, loading }: Props) {
    const [value, setValue] = useState("");
    const [exampleIndex, setExampleIndex] = useState(0);
    const submitPrompt = () => {
        const trimmed = value.trim();
        if (!trimmed || disabled)
            return;
        onSubmit(trimmed);
    };
    const handleSubmit = (e: {
        preventDefault(): void;
    }) => {
        e.preventDefault();
        submitPrompt();
    };
    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitPrompt();
        }
    };
    const applyExample = () => {
        setValue(EXAMPLES[exampleIndex]);
        setExampleIndex((i) => (i + 1) % EXAMPLES.length);
    };
    const charCount = value.length;
    const MAX_CHARS = 300;
    const tooLong = charCount > MAX_CHARS;
    return (<form className="prompt-form" onSubmit={handleSubmit} noValidate>
      <div className={`prompt-box ${disabled && !loading ? "prompt-box--disabled" : ""}`}>
        <textarea className="prompt-textarea" value={value} onChange={(e) => setValue(e.target.value)} onKeyDown={handleKeyDown} placeholder="Describe what to find…" rows={3} disabled={disabled || loading} maxLength={MAX_CHARS + 50} aria-label="Search prompt"/>

        <div className="prompt-footer">
          <button type="button" className="example-btn" onClick={applyExample} disabled={disabled || loading} tabIndex={disabled ? -1 : 0}>
            <LightbulbIcon />
            <span>Example</span>
          </button>

          <span className={`char-count ${tooLong ? "char-count--over" : ""}`}>
            {charCount}/{MAX_CHARS}
          </span>

          <button type="submit" className="submit-btn" disabled={disabled || loading || !value.trim() || tooLong} aria-label="Run query">
            {loading ? (<span className="submit-spinner" aria-hidden="true"/>) : (<SendIcon />)}
            <span>{loading ? "Running…" : "Run"}</span>
          </button>
        </div>
      </div>

      <p className="prompt-hint">
        Press <kbd>Enter</kbd> to submit · <kbd>Shift+Enter</kbd> for new line
      </p>
    </form>);
}
function SendIcon() {
    return (<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M1 7h12M8 2l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>);
}
function LightbulbIcon() {
    return (<svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
      <path d="M6.5 1a3.5 3.5 0 0 1 2 6.4V9H4.5V7.4A3.5 3.5 0 0 1 6.5 1Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
      <path d="M4.5 10.5h4M5.5 12h2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
    </svg>);
}
