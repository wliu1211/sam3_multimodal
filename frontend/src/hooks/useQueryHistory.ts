import { useState, useEffect, useCallback } from "react";
import { fetchHistory } from "../api/endpoints";
import type { QueryLogEntry } from "../types/history";
interface UseQueryHistoryReturn {
    history: QueryLogEntry[];
    loading: boolean;
    error: string | null;
    refresh: () => void;
}
const PAGE_SIZE = 20;
export function useQueryHistory(): UseQueryHistoryReturn {
    const [history, setHistory] = useState<QueryLogEntry[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetchHistory({ page: 1, pageSize: PAGE_SIZE });
            setHistory(response.entries);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load history.");
        }
        finally {
            setLoading(false);
        }
    }, []);
    useEffect(() => { load(); }, [load]);
    return { history, loading, error, refresh: load };
}
