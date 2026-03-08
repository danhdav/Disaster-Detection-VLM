import * as React from "react";
import { Link, createFileRoute } from "@tanstack/react-router";
import type { Feature } from "geojson";

import { useMapContext } from "../../../../../context/MapContext";

export const Route = createFileRoute("/(map)/disasters/$disasterId/features/$featureId")({
  component: FeatureDetailPanel,
});

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

  return (
    <>
      <Link className="panel-link" params={{ disasterId }} to="/disasters/$disasterId">
        Back to region
      </Link>

      <h2>Structure</h2>
      <div className="meta-grid">
        <span>UID</span>
        <span>{featureId}</span>
        <span>Type</span>
        <span>{String(feature?.properties?.feature_type ?? "building")}</span>
        <span>Subtype</span>
        <span>{String(feature?.properties?.subtype ?? "unclassified")}</span>
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
      ) : analysisResult ? (
        <div className="result-block">{analysisResult}</div>
      ) : (
        <p>Run analysis to request a structure-level assessment from the VLM.</p>
      )}
    </>
  );
}
