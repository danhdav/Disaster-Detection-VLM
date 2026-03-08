import * as React from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:80";

interface DebugHealthResponse {
  status: "ok";
  paths: {
    labelsExists: boolean;
    imagesExists: boolean;
  };
  counts: {
    disasters: number;
    scenes: number;
    scenesWithFeatures: number;
    labelJsonFiles: number;
    imageFiles: {
      total: number;
    };
  };
  openRouter: {
    hasApiKey: boolean;
    model: string;
  };
  errors: string[];
}

export function SystemStatus() {
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [health, setHealth] = React.useState<DebugHealthResponse | null>(null);

  const load = React.useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/debug/health`);
      if (!response.ok) {
        throw new Error(`Health check failed (${response.status})`);
      }
      setHealth((await response.json()) as DebugHealthResponse);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
      setHealth(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const issues: string[] = [];
  if (health && !health.paths.labelsExists) issues.push("labels directory not found");
  if (health && !health.paths.imagesExists) issues.push("images directory not found");
  if (health && !health.openRouter.hasApiKey) issues.push("OPENROUTER_API_KEY missing");
  if (health && health.counts.scenesWithFeatures === 0)
    issues.push("no scenes with features found");
  if (health && health.errors.length > 0)
    issues.push(`${health.errors.length} data parsing errors`);

  return (
    <section className="status-card">
      <div className="status-head">
        <h3>System Status</h3>
        <button className="status-refresh" onClick={() => void load()} type="button">
          Refresh
        </button>
      </div>

      {isLoading ? <p>Checking backend and dataset...</p> : null}
      {error ? <p className="status-error">Error: {error}</p> : null}

      {health ? (
        <>
          <div className="meta-grid">
            <span>Disasters</span>
            <span>{health.counts.disasters}</span>
            <span>Scenes</span>
            <span>{health.counts.scenes}</span>
            <span>Scenes with features</span>
            <span>{health.counts.scenesWithFeatures}</span>
            <span>Label JSONs</span>
            <span>{health.counts.labelJsonFiles}</span>
            <span>Raster images</span>
            <span>{health.counts.imageFiles.total}</span>
            <span>OpenRouter model</span>
            <span>{health.openRouter.model}</span>
          </div>

          <div className="chip-row">
            <span className={`chip ${health.paths.labelsExists ? "chip-ok" : "chip-bad"}`}>
              labels {health.paths.labelsExists ? "ok" : "missing"}
            </span>
            <span className={`chip ${health.paths.imagesExists ? "chip-ok" : "chip-bad"}`}>
              images {health.paths.imagesExists ? "ok" : "missing"}
            </span>
            <span className={`chip ${health.openRouter.hasApiKey ? "chip-ok" : "chip-bad"}`}>
              key {health.openRouter.hasApiKey ? "present" : "missing"}
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
