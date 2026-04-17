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
      "/chat": "http://localhost:4999",
      "/fire": "http://localhost:4999",
      "/image": "http://localhost:4999",
      "/analyze": "http://localhost:4999",
      "/api": "http://localhost:4999",
      "/debug": "http://localhost:4999",
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
