import { useQuery } from "@tanstack/react-query";
import { API_BASE } from "../lib/api";
import { useMemo } from "react";

interface DebugHealthResponse {
  status: "ok";
  mongodb?: {
    configured?: boolean;
    connected?: boolean;
  };
  s3?: {
    configured?: boolean;
    connected?: boolean;
  };
  s3ImagesPrefix?: string;
}

export function SystemStatus() {
  const healthQuery = useQuery<DebugHealthResponse>({
    queryKey: ["debug", "health"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/debug/health`, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`Health check failed (${response.status})`);
      }
      return (await response.json()) as DebugHealthResponse;
    },
    refetchInterval: 15_000,
    retry: false,
    refetchOnWindowFocus: false,
  });

  const health = healthQuery.data ?? null;
  const isLoading = healthQuery.isLoading;
  const error = useMemo(() => {
    if (!healthQuery.error) return null;
    return healthQuery.error instanceof Error ? healthQuery.error.message : null;
  }, [healthQuery.error]);
  const lastCheckedAt = useMemo(() => {
    if (!healthQuery.dataUpdatedAt) return null;
    return new Date(healthQuery.dataUpdatedAt);
  }, [healthQuery.dataUpdatedAt]);

  const issues: string[] = [];
  if (health && health.mongodb && health.mongodb.connected === false)
    issues.push("MongoDB not connected");
  if (health && health.s3 && health.s3.connected === false) issues.push("S3 not connected");

  return (
    <section className="status-card">
      <div className="status-head">
        <h3>System Status</h3>
        <button className="status-refresh" onClick={() => void healthQuery.refetch()} type="button">
          Refresh
        </button>
      </div>

      {isLoading ? <p>Checking backend and dataset...</p> : null}
      {error ? <p className="status-error">Error: {error}</p> : null}

      {health ? (
        <>
          <div className="meta-grid">
            <span>MongoDB configured</span>
            <span>{health.mongodb?.configured ? "yes" : "no"}</span>
            <span>MongoDB connected</span>
            <span>{health.mongodb?.connected ? "yes" : "no"}</span>
            <span>S3 configured</span>
            <span>{health.s3?.configured ? "yes" : "no"}</span>
            <span>S3 connected</span>
            <span>{health.s3?.connected ? "yes" : "no"}</span>
            <span>S3 prefix</span>
            <span>{health.s3ImagesPrefix ?? "-"}</span>
            <span>Last checked</span>
            <span>{lastCheckedAt ? lastCheckedAt.toLocaleTimeString() : "-"}</span>
          </div>

          <div className="chip-row">
            <span
              className={`chip ${health.mongodb?.connected === false ? "chip-bad" : "chip-ok"}`}
            >
              mongodb {health.mongodb?.connected === false ? "down" : "ok"}
            </span>
            <span className={`chip ${health.s3?.connected === false ? "chip-bad" : "chip-ok"}`}>
              s3 {health.s3?.connected === false ? "down" : "ok"}
            </span>
          </div>

          {issues.length > 0 ? (
            <div className="result-block">
              <strong>Action needed:</strong>
              {"\n"}
              {issues.map((issue) => `- ${issue}`).join("\n")}
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
