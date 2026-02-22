import { createFileRoute } from "@tanstack/react-router";
import "./-app.css";
export const Route = createFileRoute("/(base)/")({
  component: RouteComponent,
});

function RouteComponent() {
  return (
    <div className="content">
      <h1>Rsbuild with React</h1>
      <p>Start building amazing things with Rsbuild.</p>
    </div>
  );
}
