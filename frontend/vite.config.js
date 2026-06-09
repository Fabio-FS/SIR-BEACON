import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// For GitHub Pages project sites, set base to "/<repo-name>/".
// Change this to match your repository name before deploying.
export default defineConfig({
  plugins: [react()],
  base: "/SIR-BEACON/",
});