import { Outlet, createFileRoute, useNavigate } from "@tanstack/react-router";

import { MapView } from "../../components/MapView";
import { MapProvider, useMapContext } from "../../context/MapContext";
import "./-map.css";

export const Route = createFileRoute("/(map)")({
  component: MapLayoutRoute,
});

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
      </div>

      <aside className="floating-panel">
        <div className="panel-content">
          <Outlet />
        </div>
      </aside>
    </div>
  );
}
