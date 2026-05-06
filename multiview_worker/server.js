import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import { randomUUID } from "node:crypto";

import express from "express";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function parseCli() {
  const args = process.argv.slice(2);
  const get = (name, fallback) => {
    const index = args.indexOf(`--${name}`);
    if (index >= 0 && index + 1 < args.length) {
      return args[index + 1];
    }
    return fallback;
  };

  return {
    host: get("host", "127.0.0.1"),
    port: Number.parseInt(get("port", "3101"), 10),
  };
}

const { host, port } = parseCli();
const app = express();
app.use(express.json({ limit: "100mb" }));
app.use(express.static(path.join(__dirname, "web"), { index: false }));

let browser = null;
let page = null;
let renderChain = Promise.resolve();
let ready = false;
let ensurePagePromise = null;
const modelBlobStore = new Map();

function contentTypeForFileName(fileName) {
  const ext = path.extname(String(fileName || "").toLowerCase());
  if (ext === ".glb") return "model/gltf-binary";
  if (ext === ".gltf") return "model/gltf+json";
  if (ext === ".obj") return "text/plain; charset=utf-8";
  return "application/octet-stream";
}

function registerModelBlob(fileName, fileBuffer) {
  const token = randomUUID();
  modelBlobStore.set(token, {
    fileName,
    fileBuffer,
    expiresAt: Date.now() + 5 * 60 * 1000,
  });
  return token;
}

function cleanupExpiredModelBlobs() {
  const now = Date.now();
  for (const [token, entry] of modelBlobStore.entries()) {
    if (!entry || entry.expiresAt <= now) {
      modelBlobStore.delete(token);
    }
  }
}

function hasLivePage() {
  return Boolean(browser && page && browser.isConnected() && !page.isClosed());
}

function isTargetClosedError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("Target closed") ||
    message.includes("Session closed") ||
    message.includes("Protocol error (Runtime.callFunctionOn)") ||
    message.includes("Protocol error (Runtime.evaluate)")
  );
}

async function closeBrowserState() {
  ready = false;
  const previousPage = page;
  page = null;
  const previousBrowser = browser;
  browser = null;
  if (previousPage) {
    await previousPage.close().catch(() => {});
  }
  if (previousBrowser) {
    await previousBrowser.close().catch(() => {});
  }
}

async function createBrowserPage({ size, dpr }) {
  await closeBrowserState();

  browser = await puppeteer.launch({
    headless: "new",
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--enable-unsafe-swiftshader",
      "--use-angle=swiftshader",
    ],
  });
  page = await browser.newPage();
  page.on("console", (msg) => console.log("[multiview:page]", msg.type(), msg.text()));
  page.on("pageerror", (err) => console.error("[multiview:pageerror]", err));
  page.on("requestfailed", (req) =>
    console.error("[multiview:requestfailed]", req.url(), req.failure()?.errorText),
  );
  await page.setViewport({
    width: size,
    height: size,
    deviceScaleFactor: Math.max(1, Math.min(3, dpr)),
  });
  await page.goto(`http://${host}:${port}/multiview.html`, { waitUntil: "networkidle0" });
  await page.waitForFunction("window.__ready === true", { timeout: 30000 });
  ready = true;
}

async function ensureBrowserPage({ size, dpr }) {
  if (ensurePagePromise) {
    await ensurePagePromise;
  }

  if (hasLivePage()) {
    try {
      await page.setViewport({
        width: size,
        height: size,
        deviceScaleFactor: Math.max(1, Math.min(3, dpr)),
      });
      return page;
    } catch (error) {
      if (!isTargetClosedError(error)) {
        throw error;
      }
      await closeBrowserState();
    }
  }

  if (!ensurePagePromise) {
    ensurePagePromise = createBrowserPage({ size, dpr });
  }
  try {
    await ensurePagePromise;
  } finally {
    ensurePagePromise = null;
  }

  if (!hasLivePage()) {
    throw new Error("Browser page unavailable after initialization.");
  }
  await page.setViewport({
    width: size,
    height: size,
    deviceScaleFactor: Math.max(1, Math.min(3, dpr)),
  });
  return page;
}

async function renderMultiview(payload) {
  const size = Number.parseInt(String(payload.size ?? 512), 10) || 512;
  const dpr = Number.parseFloat(String(payload.dpr ?? 1)) || 1;
  const startedAt = Date.now();
  let result = null;

  for (let attempt = 1; attempt <= 2; attempt += 1) {
    const activePage = await ensureBrowserPage({ size, dpr });
    try {
      result = await activePage.evaluate((opts) => window.generateMultiview(opts), {
        fileName: payload.file_name,
        modelUrl: payload.model_url,
        background: payload.background ?? "#FFFFFF",
        fov: Number.parseInt(String(payload.fov ?? 35), 10) || 35,
        size,
        dpr,
        views: Number.parseInt(String(payload.views ?? 3), 10) || 3,
        skipViews: Number.parseInt(String(payload.skip_views ?? 0), 10) || 0,
        targetViewCount:
          Number.parseInt(String(payload.target_view_count ?? payload.views ?? 3), 10) || 3,
        strategy: payload.strategy ?? "fps",
        oversample: Number.parseInt(String(payload.oversample ?? 320), 10) || 320,
        radius: payload.radius ?? null,
        orbitMargin: Number.parseFloat(String(payload.orbit_margin ?? 1.35)) || 1.35,
        ensureTop: payload.ensure_top !== false,
        ditherDelayMs: Number.parseInt(String(payload.delay_ms ?? 8), 10) || 8,
      });
      break;
    } catch (error) {
      if (attempt === 1 && isTargetClosedError(error)) {
        console.warn("[multiview] target closed during render, recreating browser and retrying");
        await closeBrowserState();
        continue;
      }
      throw error;
    }
  }

  return {
    status: "ok",
    poses: result?.poses ?? 0,
    images: Array.isArray(result?.images) ? result.images : [],
    elapsed_ms: Date.now() - startedAt,
  };
}

async function warmWorker() {
  try {
    await ensureBrowserPage({ size: 512, dpr: 1 });
    console.log("[multiview] browser/page warmed");
  } catch (error) {
    console.error("[multiview] warmup failed", error);
  }
}

app.get("/health", async (_req, res) => {
  res.json({
    status: "ok",
    ready,
    browser_ready: hasLivePage(),
  });
});

app.get("/favicon.ico", (_req, res) => {
  res.status(204).end();
});

app.get("/__model/:token/:fileName", (req, res) => {
  cleanupExpiredModelBlobs();
  const token = String(req.params.token || "").trim();
  const requestedName = String(req.params.fileName || "").trim();
  const entry = modelBlobStore.get(token);
  if (!entry) {
    res.status(404).json({ detail: "Model token not found." });
    return;
  }
  if (entry.fileName !== requestedName) {
    res.status(404).json({ detail: "Model file not found for token." });
    return;
  }
  res.setHeader("Content-Type", contentTypeForFileName(entry.fileName));
  res.setHeader("Cache-Control", "no-store");
  res.send(entry.fileBuffer);
});

app.post("/render", async (req, res) => {
  const payload = req.body ?? {};
  const fileName = String(payload.file_name || "").trim();
  if (!fileName) {
    res.status(400).json({ detail: "file_name is required." });
    return;
  }
  let fileBuffer = null;
  const fileBytesBase64 = String(payload.file_bytes_base64 || "").trim();
  if (fileBytesBase64) {
    try {
      fileBuffer = Buffer.from(fileBytesBase64, "base64");
    } catch {
      res.status(400).json({ detail: "Invalid file_bytes_base64 payload." });
      return;
    }
  } else {
    const filePath = String(payload.file_path || "").trim();
    if (!filePath) {
      res.status(400).json({ detail: "file_bytes_base64 or file_path is required." });
      return;
    }
    try {
      fileBuffer = await fs.readFile(filePath);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.status(400).json({ detail: `Could not read file_path: ${message}` });
      return;
    }
  }
  if (!fileBuffer || fileBuffer.length === 0) {
    res.status(400).json({ detail: "Model payload is empty." });
    return;
  }
  cleanupExpiredModelBlobs();
  const modelToken = registerModelBlob(fileName, fileBuffer);
  const modelUrl = `http://${host}:${port}/__model/${modelToken}/${encodeURIComponent(fileName)}`;
  const normalizedPayload = {
    ...payload,
    file_name: fileName,
    model_url: modelUrl,
  };

  const job = renderChain.then(() => renderMultiview(normalizedPayload));
  renderChain = job.catch(() => {});

  try {
    const result = await job;
    res.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.status(500).json({ detail: message });
  } finally {
    modelBlobStore.delete(modelToken);
  }
});

const server = app.listen(port, host, () => {
  console.log(`[multiview] worker listening on http://${host}:${port}`);
  void warmWorker();
});

async function shutdown() {
  server.close();
  await closeBrowserState();
  process.exit(0);
}

process.on("SIGINT", () => {
  void shutdown();
});
process.on("SIGTERM", () => {
  void shutdown();
});
