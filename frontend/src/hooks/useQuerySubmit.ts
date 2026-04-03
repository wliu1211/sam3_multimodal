import { useState, useCallback } from "react";
import { submitQuery } from "../api/endpoints";
import type { QueryResponse, QueryState } from "../types/query";

interface UseQuerySubmitReturn {
  submit: (image: File, prompt: string) => Promise<void>;
  result: QueryResponse | null;
  loading: boolean;
  error: string | null;
  reset: () => void;
}

/** Readable message for any thrown/rejected value (axios quirks, strings, etc.). */
function formatSubmitError(err: unknown): string {
  if (err instanceof Error && err.message.trim()) {
    return err.message;
  }
  if (typeof err === "string" && err.trim()) {
    return err;
  }
  if (err && typeof err === "object" && "message" in err) {
    const m = (err as { message?: unknown }).message;
    if (typeof m === "string" && m.trim()) return m;
  }
  return "An unexpected error occurred. Please try again.";
}

export function useQuerySubmit(): UseQuerySubmitReturn {
  const [state, setState] = useState<QueryState>({
    status: "idle",
    result: null,
    error: null,
  });

  const submit = useCallback(async (image: File, prompt: string) => {
    setState({ status: "loading", result: null, error: null });

    try {
      const result = await submitQuery({ image, prompt });
      setState({ status: "success", result, error: null });
    } catch (err) {
      const message = formatSubmitError(err);
      setState({ status: "error", result: null, error: message });
    }
  }, []);

  const reset = useCallback(() => {
    setState({ status: "idle", result: null, error: null });
  }, []);

  return {
    submit,
    result: state.result,
    loading: state.status === "loading",
    error: state.error,
    reset,
  };
}