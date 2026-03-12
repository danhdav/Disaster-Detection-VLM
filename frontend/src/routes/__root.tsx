import * as React from "react";
import { Outlet, createRootRoute } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: RootComponent,
});

function RootComponent() {
  return (
    <div className="dark min-h-screen bg-[#0a0d10]">
      <Outlet />
    </div>
  );
}
