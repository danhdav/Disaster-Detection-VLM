import * as React from "react";
import { BitmapLayer, GeoJsonLayer } from "deck.gl";
import { DeckGL } from "@deck.gl/react";
import type { Feature, FeatureCollection, Position } from "geojson";
import Map, { NavigationControl } from "react-map-gl/maplibre";

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

function resolveImageUrl(imageUrl?: string): string | null {
  if (!imageUrl) return null;
  if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
    return imageUrl;
  }
  return imageUrl.startsWith("/") ? imageUrl : `/${imageUrl}`;
}

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

function featureToBounds(feature: Feature): Bounds | null {
  const geometry = feature.geometry;
  if (!geometry) return null;

  let minLng = Number.POSITIVE_INFINITY;
  let minLat = Number.POSITIVE_INFINITY;
  let maxLng = Number.NEGATIVE_INFINITY;
  let maxLat = Number.NEGATIVE_INFINITY;

  const visit = (coordinate: Position) => {
    const [lng, lat] = coordinate;
    minLng = Math.min(minLng, lng);
    minLat = Math.min(minLat, lat);
    maxLng = Math.max(maxLng, lng);
    maxLat = Math.max(maxLat, lat);
  };

  if (geometry.type === "Polygon") {
    for (const ring of geometry.coordinates) {
      for (const coordinate of ring) {
        visit(coordinate as Position);
      }
    }
  } else if (geometry.type === "MultiPolygon") {
    for (const polygon of geometry.coordinates) {
      for (const ring of polygon) {
        for (const coordinate of ring) {
          visit(coordinate as Position);
        }
      }
    }
  } else {
    return null;
  }

  if (!Number.isFinite(minLng)) return null;
  return [minLng, minLat, maxLng, maxLat];
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

  React.useEffect(() => {
    if (!selectedFeatureId) return;

    const selected = geoJson.features.find(
      (feature) => String(feature.properties?.uid ?? "") === selectedFeatureId,
    );
    if (!selected) return;

    const selectedBounds = featureToBounds(selected);
    if (!selectedBounds) return;

    const next = boundsToViewState(selectedBounds);
    setViewState((prev) => ({
      ...prev,
      ...next,
      // Structure focus should zoom in more than scene focus.
      zoom: Math.max(next.zoom, 16.5),
    }));
  }, [geoJson, selectedFeatureId]);

  const layers = React.useMemo(() => {
    const output: (BitmapLayer | GeoJsonLayer)[] = [];

    if (showPre && bounds) {
      const preImageUrl = resolveImageUrl(pre?.imageUrl);
      if (preImageUrl) {
        output.push(
          new BitmapLayer({
            id: "pre-image",
            image: preImageUrl,
            bounds,
            opacity: 0.7,
          }),
        );
      }
    }

    if (showPost && bounds) {
      const postImageUrl = resolveImageUrl(post?.imageUrl);
      if (postImageUrl) {
        output.push(
          new BitmapLayer({
            id: "post-image",
            image: postImageUrl,
            bounds,
            opacity: showPre ? 0.65 : 0.9,
          }),
        );
      }
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
      deviceProps={{ type: "webgl" }}
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
