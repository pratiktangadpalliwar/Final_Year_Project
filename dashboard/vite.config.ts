import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, "../server/app/static"),
    emptyOutDir: true,
  },
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    proxy: {
      "/admin": "http://localhost:8080",
      "/banks": "http://localhost:8080",
      "/metrics": "http://localhost:8080",
      "/round": "http://localhost:8080",
      "/control": "http://localhost:8080",
      "/health": "http://localhost:8080",
      "/ws":     { target: "ws://localhost:8080", ws: true },
    },
  },
});
