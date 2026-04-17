import { defineConfig } from "@rsbuild/core";
import { pluginReact } from "@rsbuild/plugin-react";
import { tanstackRouter } from "@tanstack/router-plugin/rspack";

// Docs: https://rsbuild.rs/config/
export default defineConfig({
  plugins: [pluginReact()],
  dev: {
    lazyCompilation: false,
  },
  server: {
    proxy: {
      "/chat": "http://localhost:8000",
      "/fire": "http://localhost:8000",
      "/image": "http://localhost:8000",
      "/analyze": "http://localhost:8000",
      "/api": "http://localhost:8000",
      "/debug": "http://localhost:8000",
    },
  },
  tools: {
    rspack: {
      plugins: [
        tanstackRouter({
          target: "react",
          autoCodeSplitting: true,
        }),
      ],
    },
  },
});
