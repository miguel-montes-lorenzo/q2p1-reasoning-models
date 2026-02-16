// web/server.js
import fs from "node:fs";
import path from "node:path";
import express from "express";

const app = express();

const WEB_LISTEN_HOST = "0.0.0.0";
const WEB_LISTEN_PORT = Number.parseInt(process.env.LISTEN_PORT ?? "5173", 10);

const REPO_DIR = process.env.REPO_DIR ?? "/home/root";
const API_URL_FILE = process.env.API_URL_FILE ?? "/home/root/api/ngrok-url";

app.use(express.json({ limit: "2mb" }));

const publicDir = path.resolve(REPO_DIR, "web", "public");
app.use("/", express.static(publicDir, { fallthrough: true }));

function readTextFile(filePath) {
  try {
    return fs.readFileSync(filePath, { encoding: "utf8" });
  } catch {
    return null;
  }
}

function normalizeNgrokUrl(raw) {
  const trimmed = String(raw ?? "").trim();
  if (trimmed.length === 0) return null;

  // Accept full URL as-is.
  if (/^https?:\/\//i.test(trimmed)) return trimmed;

  // If file contains only host (or host:port), assume https.
  if (/^[a-z0-9.-]+(:\d+)?$/i.test(trimmed)) return `https://${trimmed}`;

  return null;
}

function readApiUrl() {
  const candidates = [
    API_URL_FILE,
    path.resolve(REPO_DIR, "api", "ngrok-url"),
    path.resolve(REPO_DIR, "api", "ngrok_url"),
    path.resolve(REPO_DIR, "api", "ngrok-url.txt"),
    path.resolve(REPO_DIR, "api", "ngrok_url.txt"),
  ];

  for (const file of candidates) {
    const content = readTextFile(file);
    if (content == null) continue;

    const url = normalizeNgrokUrl(content);
    if (url != null) return url;
  }
  return null;
}

function readTagColors() {
  const configPath = path.resolve(REPO_DIR, "web", "tag_colors.json");
  const raw = readTextFile(configPath);
  if (raw == null) {
    return { user: "#8ecbff", think: "#e6e6e6", answer: "#b9f6ca" };
  }
  try {
    const parsed = JSON.parse(raw);
    return parsed ?? { user: "#8ecbff", think: "#e6e6e6", answer: "#b9f6ca" };
  } catch {
    return { user: "#8ecbff", think: "#e6e6e6", answer: "#b9f6ca" };
  }
}

app.get("/api/config", (_req, res) => {
  const backendUrl = readApiUrl();
  const tagColors = readTagColors();
  res.json({
    backendUrl,
    tagColors,
  });
});

async function proxyToApi(endpointPath, bodyObj) {
  const backendUrl = readApiUrl();
  if (backendUrl == null) {
    return {
      ok: false,
      status: 503,
      data: { error: "Backend URL not found. Is ./api/ngrok-url present?" },
    };
  }

  const url = `${backendUrl}${endpointPath}`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj),
  });

  const text = await resp.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { raw: text };
  }

  return { ok: resp.ok, status: resp.status, data };
}

app.post("/api/phase1/reasoning", async (req, res) => {
  try {
    const out = await proxyToApi("/phase1/reasoning", req.body ?? {});
    res.status(out.status).json(out.data);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

app.post("/api/phase2/tools", async (req, res) => {
  try {
    const out = await proxyToApi("/phase2/tools", req.body ?? {});
    res.status(out.status).json(out.data);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

app.post("/api/phase3/rag", async (req, res) => {
  try {
    const out = await proxyToApi("/phase3/rag", req.body ?? {});
    res.status(out.status).json(out.data);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

app.post("/api/phase4/agent", async (req, res) => {
  try {
    const out = await proxyToApi("/phase4/agent", req.body ?? {});
    res.status(out.status).json(out.data);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// SPA fallback
app.get("*", (_req, res) => {
  res.sendFile(path.resolve(publicDir, "index.html"));
});

app.listen(WEB_LISTEN_PORT, WEB_LISTEN_HOST, () => {
  console.log(`[web-ui] listening on http://${WEB_LISTEN_HOST}:${WEB_LISTEN_PORT}`);
  const backendUrl = readApiUrl();
  console.log(`[web-ui] backend url file: ${API_URL_FILE}`);
  console.log(`[web-ui] backend url detected: ${backendUrl ?? "null"}`);
});
