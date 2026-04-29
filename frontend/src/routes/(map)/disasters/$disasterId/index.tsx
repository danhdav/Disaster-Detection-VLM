import * as React from "react";
import { Link, createFileRoute } from "@tanstack/react-router";
import type { Feature } from "geojson";

import { useMapContext } from "../../../../context/MapContext";

export const Route = createFileRoute("/(map)/disasters/$disasterId/")({
  component: DisasterRegionPanel,
});

function DisasterRegionPanel() {
  const { disasterId } = Route.useParams();
  const {
    setActiveDisaster,
    setActiveFeature,
    sceneLabels,
    geoJson,
    isLoadingScene,
    layerMode,
    setLayerMode,
  } = useMapContext();

  React.useEffect(() => {
    void setActiveDisaster(disasterId);
  }, [disasterId, setActiveDisaster]);

  const metadata = sceneLabels?.post?.metadata ?? sceneLabels?.pre?.metadata ?? {};
  const renderValue = (value: unknown) => {
    if (typeof value === "string" || typeof value === "number") return String(value);
    return "n/a";
  };

  return (
    <>
      <Link className="panel-link" to="/">
        Back to disasters
      </Link>

      <h2 className="mono">{disasterId}</h2>
      {isLoadingScene ? <p>Loading region...</p> : null}

      <div className="chip-row" role="radiogroup" aria-label="Raster layer mode">
        <label className="chip">
          <input
            checked={layerMode === "pre"}
            name="layer-mode"
            onChange={() => setLayerMode("pre")}
            type="radio"
          />{" "}
          Pre only
        </label>
        <label className="chip">
          <input
            checked={layerMode === "post"}
            name="layer-mode"
            onChange={() => setLayerMode("post")}
            type="radio"
          />{" "}
          Post only
        </label>
        <label className="chip">
          <input
            checked={layerMode === "both"}
            name="layer-mode"
            onChange={() => setLayerMode("both")}
            type="radio"
          />{" "}
          Both
        </label>
      </div>

      <h3>Scene Metadata</h3>
      <div className="meta-grid">
        <span>Capture date</span>
        <span>{renderValue(metadata.capture_date)}</span>
        <span>Sensor</span>
        <span>{renderValue(metadata.sensor)}</span>
        <span>Disaster type</span>
        <span>{renderValue(metadata.disaster_type)}</span>
        <span>GSD</span>
        <span>{renderValue(metadata.gsd)}</span>
      </div>

      <h3>Structures</h3>
      <div className="panel-list">
        {geoJson.features.length === 0 ? (
          <p>No polygon features available for this scene.</p>
        ) : (
          geoJson.features.slice(0, 60).map((feature: Feature) => {
            const uid = String(feature.properties?.uid ?? "");
            const subtype = String(feature.properties?.subtype ?? "unclassified");
            return (
              <Link
                className="panel-link"
                key={uid}
                params={{ disasterId, featureId: uid }}
                to="/disasters/$disasterId/features/$featureId"
                onClick={() => {
                  setActiveFeature(uid);
                }}
              >
                <strong className="mono">{uid}</strong>
                <div>{subtype}</div>
              </Link>
            );
          })
        )}
      </div>
    </>
  );
}
