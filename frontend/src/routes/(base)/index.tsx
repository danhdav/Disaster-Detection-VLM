import { createFileRoute } from "@tanstack/react-router";
import { GlassButton } from "../../components/GlassButton";
import { ChatStrip } from "../../components/ChatStrip";
import "./-app.css";

export const Route = createFileRoute("/(base)/")({
  component: RouteComponent,
});

function RouteComponent() {
  return (
    <>
      <div className="hero-bg" aria-hidden />
      <div className="v1-page">
        <main className="hero">
          <span className="hero-badge">AI-powered</span>
          <h1>Disaster Detection</h1>
          <p className="hero-tagline">
            Compare before &amp; after imagery. Assess damage with AI.
          </p>
          <div className="hero-actions">
            <GlassButton
              variant="primary"
              onClick={() => console.log("[Disaster Detection] Compare before & after clicked — wire to before/after API")}
            >
              Compare before &amp; after
            </GlassButton>
            <GlassButton
              variant="secondary"
              onClick={() => console.log("[Disaster Detection] How it works clicked — wire to info/help")}
            >
              How it works
            </GlassButton>
          </div>
        </main>
        <ChatStrip />
      </div>
    </>
  );
}
