import * as React from "react";
import { Link, createFileRoute } from "@tanstack/react-router";
import type { Feature } from "geojson";

import { useMapContext } from "../../../../../context/MapContext";

export const Route = createFileRoute("/(map)/disasters/$disasterId/features/$featureId")({
  component: FeatureDetailPanel,
});

function resolveImageSrc(src: string): string {
  if (src.startsWith("http://") || src.startsWith("https://")) {
    return src;
  }
  return src.startsWith("/") ? src : `/${src}`;
}

// Fix raw blob output
function flattenAnalysisValue(value: unknown, path = ""): string[] {
  if (value === null || value === undefined) {
    return [path ? `${path}: null` : "null"];
  }

  const stringValue = JSON.stringify(value);
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return [path ? `${path}: ${stringValue}` : stringValue];
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return [path ? `${path}: []` : "[]"];
    }

    const lines: string[] = [];
    if (path) {
      lines.push(`${path}:`);
    }

    value.forEach((item, index) => {
      lines.push(...flattenAnalysisValue(item, path ? `${path}[${index}]` : `[${index}]`));
    });

    return lines;
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return [path ? `${path}: {}` : "{}"];
    }

    const lines: string[] = [];
    if (path) {
      lines.push(`${path}:`);
    }

    for (const [key, nestedValue] of entries) {
      lines.push(...flattenAnalysisValue(nestedValue, path ? `${path}.${key}` : key));
    }

    return lines;
  }

  return [path ? `${path}: ${stringValue}` : stringValue];
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
  const analysisLines = analysisResult ? flattenAnalysisValue(analysisResult) : [];

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
        {isAnalyzing ? "Running analysis..." : "Run VLM analysis"}
      </button>

      {analysisError ? (
        <div className="result-block">Error: {analysisError}</div>
      ) : analysisLines.length > 0 ? (
        <div className="result-block">
          {analysisLines.map((line, index) => (
            <div key={`${index}-${line}`}>{line}</div>
          ))}
        </div>
      ) : (
        <p>Run analysis to request a structure-level assessment from the VLM.</p>
      )}
    </>
  );
}
