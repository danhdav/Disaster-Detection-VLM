import type { FeatureCollection, Position } from "geojson";

export type Bounds = [number, number, number, number];

export function getBoundsFromFeatureCollection(fc: FeatureCollection): Bounds | null {
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

  for (const feature of fc.features) {
    if (feature.geometry.type !== "Polygon") continue;
    for (const ring of feature.geometry.coordinates) {
      for (const coordinate of ring) {
        visit(coordinate);
      }
    }
  }

  if (!Number.isFinite(minLng)) return null;
  return [minLng, minLat, maxLng, maxLat];
}
