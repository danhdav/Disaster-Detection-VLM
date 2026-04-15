import { wktToGeoJSON } from "@terraformer/wkt";

import type { Feature, FeatureCollection, Polygon } from "geojson";

export interface LabelFeature {
  properties: {
    feature_type?: string;
    subtype?: string;
    uid?: string;
    [key: string]: unknown;
  };
  wkt: string;
}

export function labelFeaturesToGeoJson(features: LabelFeature[]): FeatureCollection {
  const converted = features
    .map((feature) => {
      const geometry = wktToGeoJSON(feature.wkt) as Polygon | null;
      if (!geometry || geometry.type !== "Polygon") {
        return null;
      }
      return {
        type: "Feature",
        properties: {
          ...feature.properties,
        },
        geometry,
      } satisfies Feature;
    })
    .filter(Boolean) as Feature[];

  return {
    type: "FeatureCollection",
    features: converted,
  };
}
