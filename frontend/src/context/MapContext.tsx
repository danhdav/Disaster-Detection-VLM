import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { FeatureCollection } from "geojson";

import { API_BASE } from "../lib/api";
import { getBoundsFromFeatureCollection, type Bounds } from "../lib/bounds";
import { labelFeaturesToGeoJson, type LabelFeature } from "../lib/wkt";

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
  imageUrl?: string;
}

interface SceneLabels {
  disasterId: string;
  sceneId: string;
  bounds?: Bounds | null;
  pre?: LabelPhase | null;
  post?: LabelPhase | null;
}

interface DisastersResponse {
  disasters: DisasterSummary[];
}

interface ScenesResponse {
  scenes: SceneSummary[];
  recommendedSceneId?: string;
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
  layerMode: "pre" | "post" | "both";
  analysisResult: string | null;
  isAnalyzing: boolean;
  analysisError: string | null;
  setActiveDisaster: (disasterId: string) => Promise<void>;
  setActiveFeature: (featureId: string | null) => void;
  setLayerVisibility: (next: { showPre?: boolean; showPost?: boolean }) => void;
  setLayerMode: (mode: "pre" | "post" | "both") => void;
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
  const [activeDisasterId, setActiveDisasterIdState] = React.useState<string | null>(null);
  const [activeSceneId, setActiveSceneId] = React.useState<string | null>(null);
  const [activeFeatureId, setActiveFeatureId] = React.useState<string | null>(null);

  const [showPre, setShowPre] = React.useState(false);
  const [showPost, setShowPost] = React.useState(true);

  const disastersQuery = useQuery({
    queryKey: ["disasters"],
    queryFn: () => fetchJson<DisastersResponse>(`${API_BASE}/disasters`),
  });

  const scenesQuery = useQuery({
    queryKey: ["disaster-scenes", activeDisasterId],
    enabled: Boolean(activeDisasterId),
    queryFn: () => fetchJson<ScenesResponse>(`${API_BASE}/disasters/${activeDisasterId}/scenes`),
  });

  React.useEffect(() => {
    const scenes = scenesQuery.data?.scenes ?? [];
    if (scenes.length === 0) {
      setActiveSceneId(null);
      return;
    }

    const hasExisting =
      typeof activeSceneId === "string" && scenes.some((scene) => scene.sceneId === activeSceneId);
    if (hasExisting) return;

    const sceneToUse =
      scenesQuery.data?.recommendedSceneId ??
      scenes.find((scene) => scene.hasFeatures)?.sceneId ??
      scenes[0]?.sceneId ??
      null;

    setActiveSceneId(sceneToUse);
  }, [activeSceneId, scenesQuery.data]);

  const labelsQuery = useQuery({
    queryKey: ["scene-labels", activeDisasterId, activeSceneId],
    enabled: Boolean(activeDisasterId) && Boolean(activeSceneId),
    queryFn: () =>
      fetchJson<SceneLabels>(
        `${API_BASE}/disasters/${activeDisasterId}/scenes/${activeSceneId}/labels`,
      ),
  });

  const analyzeMutation = useMutation({
    mutationFn: async () => {
      if (!activeDisasterId || !activeSceneId) {
        throw new Error("Please select a disaster and scene first.");
      }

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

      return body.result?.text ?? "No result text returned.";
    },
  });

  const setActiveDisaster = React.useCallback(
    async (disasterId: string) => {
      setActiveDisasterIdState(disasterId);
      setActiveSceneId(null);
      setActiveFeatureId(null);
      analyzeMutation.reset();
    },
    [analyzeMutation],
  );

  const disasters = disastersQuery.data?.disasters ?? [];
  const scenes = scenesQuery.data?.scenes ?? [];
  const sceneLabels = labelsQuery.data ?? null;

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
    await analyzeMutation.mutateAsync();
  }, [analyzeMutation]);

  const analysisError =
    analyzeMutation.error instanceof Error ? analyzeMutation.error.message : null;

  const value = React.useMemo<MapContextValue>(
    () => ({
      disasters,
      scenes,
      isLoadingDisasters: disastersQuery.isLoading,
      isLoadingScene: scenesQuery.isFetching || labelsQuery.isFetching,
      activeDisasterId,
      activeSceneId,
      activeFeatureId,
      sceneLabels,
      geoJson,
      sceneBounds,
      showPre,
      showPost,
      layerMode: showPre && showPost ? "both" : showPre ? "pre" : "post",
      analysisResult: analyzeMutation.data ?? null,
      isAnalyzing: analyzeMutation.isPending,
      analysisError,
      setActiveDisaster,
      setActiveFeature: setActiveFeatureId,
      setLayerVisibility: (next) => {
        if (typeof next.showPre === "boolean") setShowPre(next.showPre);
        if (typeof next.showPost === "boolean") setShowPost(next.showPost);
      },
      setLayerMode: (mode) => {
        if (mode === "pre") {
          setShowPre(true);
          setShowPost(false);
          return;
        }
        if (mode === "post") {
          setShowPre(false);
          setShowPost(true);
          return;
        }
        setShowPre(true);
        setShowPost(true);
      },
      runAnalysis,
      clearAnalysis: () => {
        analyzeMutation.reset();
      },
    }),
    [
      disastersQuery.isLoading,
      scenesQuery.isFetching,
      labelsQuery.isFetching,
      disasters,
      scenes,
      activeDisasterId,
      activeSceneId,
      activeFeatureId,
      sceneLabels,
      geoJson,
      sceneBounds,
      showPre,
      showPost,
      analyzeMutation.data,
      analyzeMutation.isPending,
      analysisError,
      setActiveDisaster,
      analyzeMutation,
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
