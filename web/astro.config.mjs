// @ts-check
import { defineConfig } from "astro/config";

// Pure static build (no SSR adapter): the daily GitHub Action runs the ML,
// writes JSON into public/data, then `astro build` bakes a static `dist/`
// that Cloudflare serves with zero runtime code.
export default defineConfig({
  site: "https://wattcast.example.workers.dev",
  output: "static",
  build: { format: "directory" },
});
