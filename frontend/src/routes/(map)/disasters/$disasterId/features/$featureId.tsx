import * as React from "react";
import { Link, createFileRoute } from "@tanstack/react-router";
import type { Feature } from "geojson";

import { useMapContext, type AnalysisResult } from "../../../../../context/MapContext";

export const Route = createFileRoute("/(map)/disasters/$disasterId/features/$featureId")({
  component: FeatureDetailPanel,
});

function resolveImageSrc(src: string): string {
  if (src.startsWith("http://") || src.startsWith("https://")) {
    return src;
  }
  return src.startsWith("/") ? src : `/${src}`;
}

const DAMAGE_LABELS: Record<string, string> = {
  "no-damage": "No Damage",
  "minor-damage": "Minor Damage",
  "major-damage": "Major Damage",
  destroyed: "Destroyed",
};

const DAMAGE_COLORS: Record<string, string> = {
  "no-damage": "#22c55e",
  "minor-damage": "#f59e0b",
  "major-damage": "#f97316",
  destroyed: "#ef4444",
};

function DamageResultCard({ result }: { result: AnalysisResult }) {
  const level = result.damageLevel ?? "unknown";
  const label = DAMAGE_LABELS[level] ?? level;
  const color = DAMAGE_COLORS[level] ?? "#6b7280";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "12px" }}>
      <div
        style={{
          padding: "10px 14px",
          borderRadius: "8px",
          backgroundColor: `${color}22`,
          border: `2px solid ${color}`,
        }}
      >
        <span style={{ fontWeight: 700, fontSize: "1.05rem", color }}>{label}</span>
      </div>

      {result.keyEvidence && result.keyEvidence.length > 0 && (
        <div>
          <div style={{ fontWeight: 600, fontSize: "0.8rem", marginBottom: "6px", opacity: 0.7 }}>
            KEY EVIDENCE
          </div>
          <ul
            style={{
              margin: 0,
              paddingLeft: "18px",
              display: "flex",
              flexDirection: "column",
              gap: "4px",
            }}
          >
            {result.keyEvidence.map((e, i) => (
              <li key={i} style={{ fontSize: "0.875rem", lineHeight: 1.5 }}>
                {e}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function FeatureDetailPanel() {
  const { disasterId, featureId } = Route.useParams();
  const {
    setActiveDisaster,
    setActiveFeature,
    geoJson,
    runAnalysis,
    analysisResult,
    analysisError,
    isAnalyzing,
    sceneLabels,
  } = useMapContext();

  React.useEffect(() => {
    void setActiveDisaster(disasterId);
  }, [disasterId, setActiveDisaster]);

  React.useEffect(() => {
    setActiveFeature(featureId);
  }, [featureId, setActiveFeature]);

  const feature = geoJson.features.find(
    (item: Feature) => String(item.properties?.uid ?? "") === featureId,
  );

  const preImageSrc = sceneLabels?.pre?.imageUrl ?? null;
  const postImageSrc = sceneLabels?.post?.imageUrl ?? null;

  return (
    <>
      <Link className="panel-link" params={{ disasterId }} to="/disasters/$disasterId">
        Back to region
      </Link>

      <h2 className="mono">Structure</h2>
      <div className="meta-grid">
        <span>UID</span>
        <span>{featureId}</span>
        <span>Type</span>
        <span>{String(feature?.properties?.feature_type ?? "building")}</span>
        <span>Subtype</span>
        <span>{String(feature?.properties?.subtype ?? "unclassified")}</span>
      </div>

      <h3>Scene Tiles</h3>
      <div className="tile-preview-grid">
        <div className="tile-preview-card">
          <strong>Pre</strong>
          {preImageSrc ? (
            <img
              alt="Pre-disaster tile preview"
              className="tile-preview-img"
              src={resolveImageSrc(preImageSrc)}
            />
          ) : (
            <p>No pre image available.</p>
          )}
        </div>
        <div className="tile-preview-card">
          <strong>Post</strong>
          {postImageSrc ? (
            <img
              alt="Post-disaster tile preview"
              className="tile-preview-img"
              src={resolveImageSrc(postImageSrc)}
            />
          ) : (
            <p>No post image available.</p>
          )}
        </div>
      </div>

      <button
        className="panel-button"
        disabled={isAnalyzing}
        onClick={() => void runAnalysis()}
        type="button"
      >
        {isAnalyzing ? "Running VLM analysis..." : "Run VLM analysis"}
      </button>

      {analysisError ? (
        <div className="result-block">Error: {analysisError}</div>
      ) : analysisResult ? (
        <DamageResultCard result={analysisResult} />
      ) : (
        <p>Run VLM analysis to request a structure-level assessment from the language model.</p>
      )}
    </>
  );
}
