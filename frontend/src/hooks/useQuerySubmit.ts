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
      const message = err instanceof Error
        ? err.message
        : "An unexpected error occurred. Please try again.";
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