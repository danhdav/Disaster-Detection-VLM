import * as React from "react";
import { BitmapLayer, GeoJsonLayer } from "deck.gl";
import { DeckGL } from "@deck.gl/react";
import type { Feature, FeatureCollection } from "geojson";
import Map, { NavigationControl } from "react-map-gl/maplibre";

import { API_BASE } from "../lib/api";
import type { Bounds } from "../lib/bounds";

import "maplibre-gl/dist/maplibre-gl.css";

interface LabelPhase {
  imgName?: string;
  imageUrl?: string;
}

interface MapViewProps {
  geoJson: FeatureCollection;
  bounds: Bounds | null;
  pre?: LabelPhase | null;
  post?: LabelPhase | null;
  selectedFeatureId?: string | null;
  showPre: boolean;
  showPost: boolean;
  onFeatureClick?: (featureId: string) => void;
}

type ViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
};

const INITIAL_VIEW_STATE: ViewState = {
  longitude: -96,
  latitude: 37.8,
  zoom: 3,
  pitch: 0,
  bearing: 0,
};

function boundsToViewState(bounds: Bounds): Pick<ViewState, "longitude" | "latitude" | "zoom"> {
  const [minLng, minLat, maxLng, maxLat] = bounds;
  const longitude = (minLng + maxLng) / 2;
  const latitude = (minLat + maxLat) / 2;

  const lngDelta = Math.abs(maxLng - minLng);
  const latDelta = Math.abs(maxLat - minLat);
  const maxDelta = Math.max(lngDelta, latDelta, 0.0005);
  const zoom = Math.max(8, Math.min(17, Math.log2(360 / (maxDelta * 3))));

  return { longitude, latitude, zoom };
}

function fillColorForSubtype(subtype?: string): [number, number, number, number] {
  switch (subtype) {
    case "destroyed":
      return [239, 68, 68, 130];
    case "major-damage":
      return [249, 115, 22, 120];
    case "minor-damage":
      return [234, 179, 8, 120];
    case "no-damage":
      return [34, 197, 94, 110];
    default:
      return [59, 130, 246, 100];
  }
}

export function MapView({
  geoJson,
  bounds,
  pre,
  post,
  selectedFeatureId,
  showPre,
  showPost,
  onFeatureClick,
}: MapViewProps) {
  const [viewState, setViewState] = React.useState<ViewState>(INITIAL_VIEW_STATE);

  React.useEffect(() => {
    if (!bounds) return;
    const next = boundsToViewState(bounds);
    setViewState((prev) => ({ ...prev, ...next }));
  }, [bounds]);

  const layers = React.useMemo(() => {
    const output: (BitmapLayer | GeoJsonLayer)[] = [];

    if (showPre && pre?.imgName && bounds) {
      output.push(
        new BitmapLayer({
          id: "pre-image",
          image: pre.imageUrl
            ? `${API_BASE}${pre.imageUrl}`
            : `${API_BASE}/disasters/images/${pre.imgName}`,
          bounds,
          opacity: 0.7,
        }),
      );
    }

    if (showPost && post?.imgName && bounds) {
      output.push(
        new BitmapLayer({
          id: "post-image",
          image: post.imageUrl
            ? `${API_BASE}${post.imageUrl}`
            : `${API_BASE}/disasters/images/${post.imgName}`,
          bounds,
          opacity: showPre ? 0.65 : 0.9,
        }),
      );
    }

    output.push(
      new GeoJsonLayer({
        id: "structure-polygons",
        data: geoJson,
        pickable: true,
        filled: true,
        stroked: true,
        lineWidthUnits: "pixels",
        getLineWidth: (feature: Feature) =>
          feature.properties?.uid === selectedFeatureId ? 4 : 1.5,
        lineWidthMinPixels: 1,
        getLineColor: (feature: Feature) =>
          feature.properties?.uid === selectedFeatureId ? [255, 255, 255, 255] : [16, 24, 40, 220],
        getFillColor: (feature: Feature) =>
          fillColorForSubtype(feature.properties?.subtype as string),
        onClick: (info) => {
          const uid = info.object?.properties?.uid as string | undefined;
          if (uid) {
            onFeatureClick?.(uid);
          }
        },
      }),
    );

    return output;
  }, [
    bounds,
    geoJson,
    onFeatureClick,
    post?.imgName,
    pre?.imgName,
    selectedFeatureId,
    showPost,
    showPre,
  ]);

  return (
    <DeckGL
      viewState={viewState}
      controller
      onViewStateChange={(evt) => setViewState(evt.viewState as ViewState)}
      layers={layers}
      getTooltip={({ object }) => {
        if (!object) return null;
        const uid = object.properties?.uid;
        const subtype = object.properties?.subtype ?? "unclassified";
        return `Structure ${uid}\nSubtype: ${subtype}`;
      }}
    >
      <Map mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" reuseMaps>
        <NavigationControl position="top-right" />
      </Map>
    </DeckGL>
  );
}
