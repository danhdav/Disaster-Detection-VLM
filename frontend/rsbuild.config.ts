import { defineConfig } from "@rsbuild/core";
import { pluginReact } from "@rsbuild/plugin-react";
import { tanstackRouter } from "@tanstack/router-plugin/rspack";

// Docs: https://rsbuild.rs/config/
export default defineConfig({
  plugins: [pluginReact()],
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
