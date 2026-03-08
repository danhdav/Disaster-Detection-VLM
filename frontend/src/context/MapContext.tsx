import * as React from "react";
import type { FeatureCollection } from "geojson";

import { getBoundsFromFeatureCollection, type Bounds } from "../lib/bounds";
import { labelFeaturesToGeoJson, type LabelFeature } from "../lib/wkt";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:80";

export interface DisasterSummary {
  id: string;
  disasterType?: string;
  sceneCount: number;
  recommendedSceneId?: string;
  recommendedBounds?: Bounds | null;
}

export interface SceneSummary {
  sceneId: string;
  hasFeatures: boolean;
  bounds?: Bounds | null;
  hasPre: boolean;
  hasPost: boolean;
}

interface LabelPhase {
  metadata: Record<string, unknown>;
  features: {
    lng_lat: LabelFeature[];
    xy: unknown[];
  };
  imgName?: string;
}

interface SceneLabels {
  disasterId: string;
  sceneId: string;
  bounds?: Bounds | null;
  pre?: LabelPhase | null;
  post?: LabelPhase | null;
}

interface AnalyzeResponse {
  status: "ok" | "error";
  result?: {
    text: string;
    model?: string;
  };
  error?: string;
}

interface MapContextValue {
  disasters: DisasterSummary[];
  scenes: SceneSummary[];
  isLoadingDisasters: boolean;
  isLoadingScene: boolean;
  activeDisasterId: string | null;
  activeSceneId: string | null;
  activeFeatureId: string | null;
  sceneLabels: SceneLabels | null;
  geoJson: FeatureCollection;
  sceneBounds: Bounds | null;
  showPre: boolean;
  showPost: boolean;
  analysisResult: string | null;
  isAnalyzing: boolean;
  analysisError: string | null;
  setActiveDisaster: (disasterId: string) => Promise<void>;
  setActiveFeature: (featureId: string | null) => void;
  setLayerVisibility: (next: { showPre?: boolean; showPost?: boolean }) => void;
  runAnalysis: () => Promise<void>;
  clearAnalysis: () => void;
}

const MapContext = React.createContext<MapContextValue | null>(null);

function emptyFeatureCollection(): FeatureCollection {
  return {
    type: "FeatureCollection",
    features: [],
  };
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed request (${response.status}): ${url}`);
  }
  return (await response.json()) as T;
}

function chooseFeatures(labels: SceneLabels | null): LabelFeature[] {
  if (!labels) return [];
  const post = labels.post?.features.lng_lat ?? [];
  if (post.length > 0) return post;
  return labels.pre?.features.lng_lat ?? [];
}

export function MapProvider({ children }: { children: React.ReactNode }) {
  const [disasters, setDisasters] = React.useState<DisasterSummary[]>([]);
  const [scenes, setScenes] = React.useState<SceneSummary[]>([]);
  const [isLoadingDisasters, setIsLoadingDisasters] = React.useState(true);
  const [isLoadingScene, setIsLoadingScene] = React.useState(false);

  const [activeDisasterId, setActiveDisasterIdState] = React.useState<string | null>(null);
  const [activeSceneId, setActiveSceneId] = React.useState<string | null>(null);
  const [activeFeatureId, setActiveFeatureId] = React.useState<string | null>(null);

  const [showPre, setShowPre] = React.useState(false);
  const [showPost, setShowPost] = React.useState(true);

  const [sceneLabels, setSceneLabels] = React.useState<SceneLabels | null>(null);

  const [analysisResult, setAnalysisResult] = React.useState<string | null>(null);
  const [analysisError, setAnalysisError] = React.useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;

    const loadDisasters = async () => {
      try {
        setIsLoadingDisasters(true);
        const response = await fetchJson<{ disasters: DisasterSummary[] }>(`${API_BASE}/disasters`);
        if (!cancelled) {
          setDisasters(response.disasters);
        }
      } catch (error) {
        // Keep UI usable even when backend isn't reachable.
        if (!cancelled) {
          setDisasters([]);
        }
        console.error(error);
      } finally {
        if (!cancelled) {
          setIsLoadingDisasters(false);
        }
      }
    };

    void loadDisasters();
    return () => {
      cancelled = true;
    };
  }, []);

  const setActiveDisaster = React.useCallback(async (disasterId: string) => {
    setActiveDisasterIdState(disasterId);
    setActiveFeatureId(null);
    setAnalysisResult(null);
    setAnalysisError(null);
    setIsLoadingScene(true);

    try {
      const scenesResponse = await fetchJson<{
        scenes: SceneSummary[];
        recommendedSceneId?: string;
      }>(`${API_BASE}/disasters/${disasterId}/scenes`);

      setScenes(scenesResponse.scenes);

      const sceneToUse =
        scenesResponse.recommendedSceneId ??
        scenesResponse.scenes.find((scene) => scene.hasFeatures)?.sceneId ??
        scenesResponse.scenes[0]?.sceneId;

      if (!sceneToUse) {
        setSceneLabels(null);
        setActiveSceneId(null);
        return;
      }

      setActiveSceneId(sceneToUse);
      const labels = await fetchJson<SceneLabels>(
        `${API_BASE}/disasters/${disasterId}/scenes/${sceneToUse}/labels`,
      );
      setSceneLabels(labels);
    } finally {
      setIsLoadingScene(false);
    }
  }, []);

  const geoJson = React.useMemo(() => {
    const features = chooseFeatures(sceneLabels);
    if (features.length === 0) return emptyFeatureCollection();
    return labelFeaturesToGeoJson(features);
  }, [sceneLabels]);

  const sceneBounds = React.useMemo(() => {
    if (sceneLabels?.bounds && sceneLabels.bounds.length === 4) {
      return sceneLabels.bounds as Bounds;
    }
    return getBoundsFromFeatureCollection(geoJson);
  }, [geoJson, sceneLabels]);

  const runAnalysis = React.useCallback(async () => {
    if (!activeDisasterId || !activeSceneId) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setAnalysisError(null);
    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          disasterId: activeDisasterId,
          sceneId: activeSceneId,
          featureId: activeFeatureId,
        }),
      });

      const body = (await response.json()) as AnalyzeResponse;
      if (!response.ok || body.status !== "ok") {
        throw new Error(body.error ?? `Analysis failed (${response.status})`);
      }

      setAnalysisResult(body.result?.text ?? "No result text returned.");
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsAnalyzing(false);
    }
  }, [activeDisasterId, activeFeatureId, activeSceneId]);

  const value = React.useMemo<MapContextValue>(
    () => ({
      disasters,
      scenes,
      isLoadingDisasters,
      isLoadingScene,
      activeDisasterId,
      activeSceneId,
      activeFeatureId,
      sceneLabels,
      geoJson,
      sceneBounds,
      showPre,
      showPost,
      analysisResult,
      isAnalyzing,
      analysisError,
      setActiveDisaster,
      setActiveFeature: setActiveFeatureId,
      setLayerVisibility: (next) => {
        if (typeof next.showPre === "boolean") setShowPre(next.showPre);
        if (typeof next.showPost === "boolean") setShowPost(next.showPost);
      },
      runAnalysis,
      clearAnalysis: () => {
        setAnalysisResult(null);
        setAnalysisError(null);
      },
    }),
    [
      disasters,
      scenes,
      isLoadingDisasters,
      isLoadingScene,
      activeDisasterId,
      activeSceneId,
      activeFeatureId,
      sceneLabels,
      geoJson,
      sceneBounds,
      showPre,
      showPost,
      analysisResult,
      isAnalyzing,
      analysisError,
      setActiveDisaster,
      runAnalysis,
    ],
  );

  return <MapContext.Provider value={value}>{children}</MapContext.Provider>;
}

export function useMapContext() {
  const context = React.useContext(MapContext);
  if (!context) {
    throw new Error("useMapContext must be used within a MapProvider");
  }
  return context;
}
