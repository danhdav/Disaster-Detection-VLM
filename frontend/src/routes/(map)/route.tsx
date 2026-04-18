import { CatchBoundary, Outlet, createFileRoute, useNavigate } from "@tanstack/react-router";

import { MapView } from "../../components/MapView";
import { ChatSidebar } from "../../components/ChatSidebar/ChatSidebar";
import { SystemStatus } from "../../components/SystemStatus";
import { MapProvider, useMapContext } from "../../context/MapContext";
import "./-map.css";

export const Route = createFileRoute("/(map)")({
  component: MapLayoutRoute,
});

function MapErrorFallback({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div className="result-block">
      Unable to initialize map rendering on this device.
      {error.message ? ` ${error.message}` : ""}
      <div>
        <button className="panel-button" onClick={reset} type="button">
          Retry map
        </button>
      </div>
    </div>
  );
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
        <CatchBoundary
          errorComponent={MapErrorFallback}
          getResetKey={() => `${activeDisasterId ?? "none"}:${activeFeatureId ?? "none"}`}
          onCatch={() => {
            // Keep map failures isolated so panel routes remain usable.
          }}
        >
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
        </CatchBoundary>
      </div>

      <aside className="floating-panel">
        <div className="panel-content">
          <SystemStatus />
          <Outlet />
        </div>
      </aside>

      <aside className="chat-sidebar-panel" aria-label="Disaster assessment chat sidebar">
        <ChatSidebar />
      </aside>
    </div>
  );
}
