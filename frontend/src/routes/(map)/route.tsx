import * as React from "react";
import { Outlet, createFileRoute, useNavigate } from "@tanstack/react-router";

import { MapView } from "../../components/MapView";
import { SystemStatus } from "../../components/SystemStatus";
import { MapProvider, useMapContext } from "../../context/MapContext";
import "./-map.css";

export const Route = createFileRoute("/(map)")({
  component: MapLayoutRoute,
});

class MapErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; message: string | null }
> {
  public constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, message: null };
  }

  public static getDerivedStateFromError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return { hasError: true, message };
  }

  public componentDidCatch() {
    // Keep map failures isolated so panel routes remain usable.
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="result-block">
          Unable to initialize map rendering on this device.
          {this.state.message ? ` ${this.state.message}` : ""}
        </div>
      );
    }
    return this.props.children;
  }
}

function MapLayoutRoute() {
  return (
    <MapProvider>
      <MapLayout />
    </MapProvider>
  );
}

function MapLayout() {
  const navigate = useNavigate();
  const {
    activeDisasterId,
    geoJson,
    sceneBounds,
    sceneLabels,
    activeFeatureId,
    showPre,
    showPost,
  } = useMapContext();

  return (
    <div className="map-layout">
      <div className="map-canvas">
        <MapErrorBoundary>
          <MapView
            geoJson={geoJson}
            bounds={sceneBounds}
            pre={sceneLabels?.pre}
            post={sceneLabels?.post}
            selectedFeatureId={activeFeatureId}
            showPre={showPre}
            showPost={showPost}
            onFeatureClick={(featureId) => {
              if (!activeDisasterId) return;
              void navigate({
                to: "/disasters/$disasterId/features/$featureId",
                params: { disasterId: activeDisasterId, featureId },
              });
            }}
          />
        </MapErrorBoundary>
      </div>

      <aside className="floating-panel">
        <div className="panel-content">
          <SystemStatus />
          <Outlet />
        </div>
      </aside>
    </div>
  );
}
