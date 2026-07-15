const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const readline = require("node:readline");

let mainWindow;
let python;
let sequence = 0;
const pending = new Map();

function repositoryRoot() {
  if (!app.isPackaged) return path.resolve(__dirname, "../..");
  return process.resourcesPath;
}

function pythonExecutable() {
  const bundled = path.join(process.resourcesPath, "python-bridge", "latticezero-bridge");
  const candidates = [
    process.env.LATTICEZERO_PYTHON,
    app.isPackaged && fs.existsSync(bundled) ? bundled : null,
    path.join(repositoryRoot(), ".venv", "bin", "python"),
    "python3",
    "python",
  ].filter(Boolean);
  return candidates.find((candidate) => {
    if (!candidate.includes(path.sep)) return true;
    return fs.existsSync(candidate);
  });
}

function bridgeCommand() {
  const bundled = path.join(process.resourcesPath, "python-bridge", "latticezero-bridge");
  if (app.isPackaged && fs.existsSync(bundled)) return { executable: bundled, args: [] };
  const script = app.isPackaged
    ? path.join(process.resourcesPath, "python-bridge", "bridge.py")
    : path.join(__dirname, "../python/bridge.py");
  return { executable: pythonExecutable(), args: [script] };
}

function startBridge() {
  const command = bridgeCommand();
  python = spawn(command.executable, command.args, {
    cwd: repositoryRoot(),
    env: {
      ...process.env,
      PYTHONPATH: [repositoryRoot(), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
      PYTHONUNBUFFERED: "1",
    },
    stdio: ["pipe", "pipe", "pipe"],
  });

  readline.createInterface({ input: python.stdout }).on("line", (line) => {
    try {
      const message = JSON.parse(line);
      if (message.event) {
        mainWindow?.webContents.send("cadflow:event", message);
        return;
      }
      const waiter = pending.get(message.id);
      if (waiter) {
        pending.delete(message.id);
        message.error ? waiter.reject(new Error(message.error)) : waiter.resolve(message.result);
      }
    } catch (error) {
      mainWindow?.webContents.send("cadflow:event", {
        event: "bridge.log",
        payload: { level: "error", message: `Invalid bridge output: ${line}` },
      });
    }
  });

  python.stderr.on("data", (chunk) => {
    mainWindow?.webContents.send("cadflow:event", {
      event: "bridge.log",
      payload: { level: "error", message: chunk.toString() },
    });
  });

  python.on("exit", (code) => {
    for (const waiter of pending.values()) waiter.reject(new Error(`Python bridge exited (${code})`));
    pending.clear();
    mainWindow?.webContents.send("cadflow:event", {
      event: "bridge.status",
      payload: { connected: false, code },
    });
  });
}

function request(method, params = {}) {
  return new Promise((resolve, reject) => {
    if (!python || python.killed) startBridge();
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    python.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error(`${method} timed out`));
      }
    }, method === "run_pipeline" || method === "run_autopilot" ? 300000 : 30000);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1520,
    height: 960,
    minWidth: 1180,
    minHeight: 720,
    backgroundColor: "#080b10",
    titleBarStyle: "hidden",
    titleBarOverlay: { color: "#080b10", symbolColor: "#9ca7b4", height: 38 },
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else if (!app.isPackaged) {
    mainWindow.loadURL("http://127.0.0.1:5173");
  } else {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

app.whenReady().then(() => {
  ipcMain.handle("cadflow:request", (_, method, params) => request(method, params));
  ipcMain.handle("dialog:directory", async () => {
    const result = await dialog.showOpenDialog(mainWindow, { properties: ["openDirectory", "createDirectory"] });
    return result.canceled ? null : result.filePaths[0];
  });
  ipcMain.handle("shell:reveal", (_, filePath) => shell.showItemInFolder(filePath));
  ipcMain.handle("shell:open", (_, target) => shell.openPath(target));
  createWindow();
  startBridge();
});

app.on("window-all-closed", () => {
  python?.kill();
  if (process.platform !== "darwin") app.quit();
});
