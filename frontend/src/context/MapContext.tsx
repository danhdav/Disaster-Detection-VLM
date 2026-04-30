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

interface FireLabelDocument {
  metadata?: {
    disaster?: string;
    disaster_type?: string;
    img_name?: string;
    [key: string]: unknown;
  };
  features?: {
    lng_lat?: LabelFeature[];
    xy?: unknown[];
  };
}

export interface AnalysisResult {
  damageLevel: string;
  confidence: string;
  keyEvidence: string[];
  model?: string;
  sceneId?: string;
  featureId?: string | null;
}

export interface CnnAnalysisResult {
  damageLevel: string;
  confidence: string;
  probabilities: Record<string, number>;
  model?: string;
  sceneId?: string;
  featureId?: string | null;
}

interface AnalyzeResponse {
  status: "ok" | "error";
  result?: AnalysisResult;
  error?: string;
}

interface CnnAnalyzeResponse {
  status: "ok" | "error";
  result?: CnnAnalysisResult;
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
  analysisResult: AnalysisResult | null;
  isAnalyzing: boolean;
  analysisError: string | null;
  cnnResult: CnnAnalysisResult | null;
  isCnnAnalyzing: boolean;
  cnnError: string | null;
  setActiveDisaster: (disasterId: string) => Promise<void>;
  setActiveFeature: (featureId: string | null) => void;
  setLayerVisibility: (next: { showPre?: boolean; showPost?: boolean }) => void;
  setLayerMode: (mode: "pre" | "post" | "both") => void;
  runAnalysis: () => Promise<void>;
  clearAnalysis: () => void;
  runCnnAnalysis: () => Promise<void>;
  clearCnnAnalysis: () => void;
}

const MapContext = React.createContext<MapContextValue | null>(null);

function emptyFeatureCollection(): FeatureCollection {
  return {
    type: "FeatureCollection",
    features: [],
  };
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
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

type ScenePhase = "pre" | "post" | null;

function parseSceneInfoFromImgName(
  imgName?: string,
): { sceneId: string; phase: ScenePhase } | null {
  if (!imgName) return null;

  const fileName = imgName.split("/").pop()?.trim();
  if (!fileName) return null;

  const match = fileName.match(/^(.*)_(pre|post)_disaster(?:\.[^.]+)?$/i);
  if (match) {
    return {
      sceneId: match[1],
      phase: match[2].toLowerCase() as "pre" | "post",
    };
  }

  const withoutExtension = fileName.replace(/\.[^.]+$/, "");
  if (!withoutExtension) return null;

  return { sceneId: withoutExtension, phase: null };
}

function phaseFromDoc(doc: FireLabelDocument | undefined, imageUrl?: string): LabelPhase | null {
  if (!doc) return null;
  return {
    metadata: doc.metadata ?? {},
    features: {
      lng_lat: doc.features?.lng_lat ?? [],
      xy: doc.features?.xy ?? [],
    },
    imgName: doc.metadata?.img_name,
    imageUrl,
  };
}

export function MapProvider({ children }: { children: React.ReactNode }) {
  const [activeDisasterId, setActiveDisasterIdState] = React.useState<string | null>(null);
  const [activeSceneId, setActiveSceneId] = React.useState<string | null>(null);
  const [activeFeatureId, setActiveFeatureId] = React.useState<string | null>(null);

  const [showPre, setShowPre] = React.useState(false);
  const [showPost, setShowPost] = React.useState(true);

  const fireLabelsQuery = useQuery({
    queryKey: ["fire"],
    queryFn: () => fetchJson<FireLabelDocument[]>(`${API_BASE}/fire`),
    retry: false,
    staleTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });

  const disasters = React.useMemo<DisasterSummary[]>(() => {
    const docs = fireLabelsQuery.data ?? [];
    const byDisaster = new Map<
      string,
      {
        disasterType?: string;
        scenes: Map<string, { hasFeatures: boolean }>;
      }
    >();

    for (const doc of docs) {
      const disasterId = doc.metadata?.disaster;
      const sceneInfo = parseSceneInfoFromImgName(doc.metadata?.img_name);
      const sceneId = sceneInfo?.sceneId ?? null;
      if (!disasterId || !sceneId) continue;

      const existing = byDisaster.get(disasterId) ?? {
        disasterType: doc.metadata?.disaster_type,
        scenes: new Map<string, { hasFeatures: boolean }>(),
      };
      const hasFeatures = (doc.features?.lng_lat ?? []).length > 0;
      const currentScene = existing.scenes.get(sceneId) ?? {
        hasFeatures: false,
      };
      existing.scenes.set(sceneId, {
        hasFeatures: currentScene.hasFeatures || hasFeatures,
      });
      byDisaster.set(disasterId, existing);
    }

    return Array.from(byDisaster.entries()).map(([id, value]) => {
      const scenes = Array.from(value.scenes.entries());
      const featuredSceneId = scenes.find(([, scene]) => scene.hasFeatures)?.[0];

      return {
        id,
        disasterType: value.disasterType,
        sceneCount: scenes.length,
        // Prefer a scene with labels so the map can compute bounds and focus immediately.
        recommendedSceneId: featuredSceneId ?? scenes[0]?.[0] ?? undefined,
        recommendedBounds: null,
      };
    });
  }, [fireLabelsQuery.data]);

  const scenes = React.useMemo<SceneSummary[]>(() => {
    const docs = fireLabelsQuery.data ?? [];
    if (!activeDisasterId) return [];

    const byScene = new Map<
      string,
      {
        pre?: FireLabelDocument;
        post?: FireLabelDocument;
      }
    >();

    for (const doc of docs) {
      if (doc.metadata?.disaster !== activeDisasterId) continue;
      const sceneInfo = parseSceneInfoFromImgName(doc.metadata?.img_name);
      const sceneId = sceneInfo?.sceneId ?? null;
      if (!sceneId) continue;

      const existing = byScene.get(sceneId) ?? {};
      if (sceneInfo?.phase === "pre") {
        existing.pre = doc;
      } else if (sceneInfo?.phase === "post") {
        existing.post = doc;
      } else if (!existing.post) {
        // If phase is unknown, keep one doc so the scene still appears.
        existing.post = doc;
      }
      byScene.set(sceneId, existing);
    }

    return Array.from(byScene.entries()).map(([sceneId, pair]) => {
      const preFeatures = pair.pre?.features?.lng_lat ?? [];
      const postFeatures = pair.post?.features?.lng_lat ?? [];
      const hasFeatures = preFeatures.length > 0 || postFeatures.length > 0;

      let bounds: Bounds | null = null;
      const chosen = postFeatures.length > 0 ? postFeatures : preFeatures;
      if (chosen.length > 0) {
        bounds = getBoundsFromFeatureCollection(labelFeaturesToGeoJson(chosen));
      }

      return {
        sceneId,
        hasFeatures,
        bounds,
        hasPre: Boolean(pair.pre),
        hasPost: Boolean(pair.post),
      };
    });
  }, [activeDisasterId, fireLabelsQuery.data]);

  React.useEffect(() => {
    const availableScenes = scenes;
    if (availableScenes.length === 0) {
      setActiveSceneId(null);
      return;
    }

    const hasExisting =
      typeof activeSceneId === "string" &&
      availableScenes.some((scene) => scene.sceneId === activeSceneId);
    if (hasExisting) return;

    const sceneToUse =
      availableScenes.find((scene) => scene.hasFeatures)?.sceneId ??
      availableScenes[0]?.sceneId ??
      null;

    setActiveSceneId(sceneToUse);
  }, [activeSceneId, scenes]);

  const {
    mutateAsync: analyzeAsync,
    reset: resetAnalysisMutation,
    data: analysisMutationData,
    isPending: isAnalyzing,
    error: analysisMutationError,
  } = useMutation({
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

      const r = body.result;
      if (!r) throw new Error("No result returned from analysis.");
      return r;
    },
  });

  const {
    mutateAsync: cnnAnalyzeAsync,
    reset: resetCnnMutation,
    data: cnnMutationData,
    isPending: isCnnAnalyzing,
    error: cnnMutationError,
  } = useMutation({
    mutationFn: async () => {
      if (!activeDisasterId || !activeSceneId) {
        throw new Error("Please select a disaster and scene first.");
      }
      if (!activeFeatureId) {
        throw new Error("Please select a building to run CNN analysis.");
      }

      const response = await fetch(`${API_BASE}/cnn-analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disasterId: activeDisasterId,
          sceneId: activeSceneId,
          featureId: activeFeatureId,
        }),
      });

      const body = (await response.json()) as CnnAnalyzeResponse;
      if (!response.ok || body.status !== "ok") {
        throw new Error(body.error ?? `CNN analysis failed (${response.status})`);
      }

      const r = body.result;
      if (!r) throw new Error("No result returned from CNN analysis.");
      return r;
    },
  });

  const setActiveDisaster = React.useCallback(
    async (disasterId: string) => {
      if (activeDisasterId === disasterId) return;
      setActiveDisasterIdState(disasterId);
      setActiveSceneId(null);
      setActiveFeatureId(null);
      resetAnalysisMutation();
      resetCnnMutation();
    },
    [activeDisasterId, resetAnalysisMutation, resetCnnMutation],
  );

  const sceneLabels = React.useMemo<SceneLabels | null>(() => {
    if (!activeDisasterId || !activeSceneId) return null;

    const docs = fireLabelsQuery.data ?? [];
    const preDoc = docs.find(
      (doc) =>
        doc.metadata?.disaster === activeDisasterId &&
        parseSceneInfoFromImgName(doc.metadata?.img_name)?.sceneId === activeSceneId &&
        parseSceneInfoFromImgName(doc.metadata?.img_name)?.phase === "pre",
    );
    const postDoc = docs.find(
      (doc) =>
        doc.metadata?.disaster === activeDisasterId &&
        parseSceneInfoFromImgName(doc.metadata?.img_name)?.sceneId === activeSceneId &&
        parseSceneInfoFromImgName(doc.metadata?.img_name)?.phase !== "pre",
    );

    const pre = phaseFromDoc(
      preDoc,
      preDoc ? API_BASE + `/image/${encodeURIComponent(activeSceneId)}/pre` : undefined,
    );
    const post = phaseFromDoc(
      postDoc,
      postDoc ? API_BASE + `/image/${encodeURIComponent(activeSceneId)}/post` : undefined,
    );

    const chosen = (post?.features.lng_lat ?? []).length > 0 ? post : pre;
    const bounds =
      chosen && chosen.features.lng_lat.length > 0
        ? getBoundsFromFeatureCollection(labelFeaturesToGeoJson(chosen.features.lng_lat))
        : null;

    return {
      disasterId: activeDisasterId,
      sceneId: activeSceneId,
      bounds,
      pre,
      post,
    };
  }, [activeDisasterId, activeSceneId, fireLabelsQuery.data]);

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
    await analyzeAsync();
  }, [analyzeAsync]);

  const clearAnalysis = React.useCallback(() => {
    resetAnalysisMutation();
  }, [resetAnalysisMutation]);

  const runCnnAnalysis = React.useCallback(async () => {
    await cnnAnalyzeAsync();
  }, [cnnAnalyzeAsync]);

  const clearCnnAnalysis = React.useCallback(() => {
    resetCnnMutation();
  }, [resetCnnMutation]);

  const setActiveFeature = React.useCallback(
    (featureId: string | null) => {
      setActiveFeatureId(featureId);
      resetAnalysisMutation();
    },
    [resetAnalysisMutation],
  );

  const analysisError =
    analysisMutationError instanceof Error ? analysisMutationError.message : null;

  const cnnError =
    cnnMutationError instanceof Error ? cnnMutationError.message : null;

  const value = React.useMemo<MapContextValue>(
    () => ({
      disasters,
      scenes,
      isLoadingDisasters: fireLabelsQuery.isLoading,
      isLoadingScene: false,
      activeDisasterId,
      activeSceneId,
      activeFeatureId,
      sceneLabels,
      geoJson,
      sceneBounds,
      showPre,
      showPost,
      layerMode: showPre && showPost ? "both" : showPre ? "pre" : "post",
      analysisResult: analysisMutationData ?? null,
      isAnalyzing,
      analysisError,
      cnnResult: cnnMutationData ?? null,
      isCnnAnalyzing,
      cnnError,
      setActiveDisaster,
      setActiveFeature,
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
      clearAnalysis,
      runCnnAnalysis,
      clearCnnAnalysis,
    }),
    [
      fireLabelsQuery.isLoading,
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
      analysisMutationData,
      isAnalyzing,
      analysisError,
      cnnMutationData,
      isCnnAnalyzing,
      cnnError,
      setActiveDisaster,
      setActiveFeature,
      clearAnalysis,
      runAnalysis,
      runCnnAnalysis,
      clearCnnAnalysis,
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
