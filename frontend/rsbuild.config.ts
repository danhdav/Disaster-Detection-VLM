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
      "/fire": "http://localhost:5000",
      "/image": "http://localhost:5000",
      "/analyze": "http://localhost:5000",
      "/api": "http://localhost:5000",
      "/debug": "http://localhost:5000",
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
