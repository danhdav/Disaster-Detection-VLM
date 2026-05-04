import { Link, createFileRoute } from "@tanstack/react-router";

import { useMapContext } from "../../context/MapContext";

export const Route = createFileRoute("/(map)/")({
  component: DisasterListPanel,
});

function DisasterListPanel() {
  const { disasters, isLoadingDisasters, setActiveDisaster } = useMapContext();

  return (
    <>
      <h2>Disasters</h2>
      <p>Select a disaster to focus on a scene with labeled structures.</p>

      {isLoadingDisasters ? (
        <p>Loading disasters...</p>
      ) : (
        <div className="panel-list">
          {disasters.map((disaster) => (
            <Link
              className="panel-link"
              key={disaster.id}
              to="/disasters/$disasterId"
              params={{ disasterId: disaster.id }}
              onClick={() => {
                void setActiveDisaster(disaster.id);
              }}
            >
              <strong className="mono">{disaster.id}</strong>
              <div>{disaster.disasterType ?? "unknown type"}</div>
              <div className="mono">{disaster.sceneCount} scenes</div>
            </Link>
          ))}
        </div>
      )}
    </>
  );
}
