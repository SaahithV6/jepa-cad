const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");
const readline = require("node:readline");

let mainWindow;
let python;
let sequence = 0;
const pending = new Map();

function canWriteDir(dir) {
  try {
    fs.mkdirSync(dir, { recursive: true });
    const probe = path.join(dir, `.lz-write-${process.pid}`);
    fs.writeFileSync(probe, "ok");
    fs.unlinkSync(probe);
    return true;
  } catch {
    return false;
  }
}

function resolveWritableDir(candidates) {
  for (const dir of candidates) {
    if (canWriteDir(dir)) return dir;
  }
  return candidates[candidates.length - 1];
}

/** Must run before app.ready — root-owned ~/.config/latticezero crashes Chromium. */
function configureUserData() {
  const home = os.homedir();
  const uid = typeof process.getuid === "function" ? process.getuid() : "u";
  const chosen = resolveWritableDir([
    path.join(home, ".config", "latticezero"),
    path.join(home, ".local", "share", "latticezero", "electron"),
    path.join(os.tmpdir(), `latticezero-electron-${uid}`),
  ]);
  app.setPath("userData", chosen);
  return chosen;
}

/** AppImages ship chrome-sandbox without setuid → Zygote fatal without this. */
function configureLinuxSandbox() {
  if (process.platform !== "linux") return;
  const candidates = [
    path.join(path.dirname(process.execPath), "chrome-sandbox"),
    path.join(process.resourcesPath || "", "..", "chrome-sandbox"),
  ];
  let usable = false;
  for (const sandboxPath of candidates) {
    try {
      const st = fs.statSync(sandboxPath);
      if ((st.mode & fs.constants.S_ISUID) !== 0 && st.uid === 0) {
        usable = true;
        break;
      }
    } catch {
      /* missing */
    }
  }
  if (!usable) {
    app.commandLine.appendSwitch("no-sandbox");
    app.commandLine.appendSwitch("disable-setuid-sandbox");
    app.commandLine.appendSwitch("disable-gpu-sandbox");
  }
}

function resolveDataDir() {
  const home = os.homedir();
  const uid = typeof process.getuid === "function" ? process.getuid() : "u";
  return resolveWritableDir([
    process.env.LATTICEZERO_DATA_DIR,
    path.join(home, ".local", "share", "latticezero"),
    path.join(home, ".local", "share", "latticezero-user"),
    path.join(os.tmpdir(), `latticezero-data-${uid}`),
  ].filter(Boolean));
}

const userDataPath = configureUserData();
configureLinuxSandbox();

function repositoryRoot() {
  if (!app.isPackaged) return path.resolve(__dirname, "../..");
  return process.resourcesPath;
}

function pythonExecutable() {
  const bundled = path.join(process.resourcesPath || "", "python-bridge", "latticezero-bridge");
  const candidates = [
    process.env.LATTICEZERO_PYTHON,
    app.isPackaged && fs.existsSync(bundled) ? bundled : null,
    path.join(repositoryRoot(), ".venv", "bin", "python"),
    "/usr/bin/python3",
    "python3",
    "python",
  ].filter(Boolean);
  return candidates.find((candidate) => {
    if (!candidate.includes(path.sep)) return true;
    return fs.existsSync(candidate);
  });
}

function bridgeCommand() {
  const bundled = path.join(process.resourcesPath || "", "python-bridge", "latticezero-bridge");
  if (app.isPackaged && fs.existsSync(bundled)) return { executable: bundled, args: [] };
  const script = app.isPackaged
    ? path.join(process.resourcesPath, "python-bridge", "bridge.py")
    : path.join(__dirname, "../python/bridge.py");
  const executable = pythonExecutable();
  if (!executable) throw new Error("No Python interpreter found for LatticeZero bridge");
  return { executable, args: [script] };
}

function startBridge() {
  try {
    const command = bridgeCommand();
    const repoRoot = repositoryRoot();
    const dataDir = resolveDataDir();
    python = spawn(command.executable, command.args, {
      cwd: repoRoot,
      env: {
        ...process.env,
        LATTICEZERO_REPO_ROOT: process.env.LATTICEZERO_REPO_ROOT || repoRoot,
        LATTICEZERO_DATA_DIR: dataDir,
        PYTHONPATH: [repoRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (error) {
    mainWindow?.webContents.send("cadflow:event", {
      event: "bridge.status",
      payload: { connected: false, message: String(error) },
    });
    return;
  }

  python.on("error", (error) => {
    mainWindow?.webContents.send("cadflow:event", {
      event: "bridge.status",
      payload: { connected: false, message: String(error) },
    });
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
    } catch {
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
    if (!python || python.killed || !python.stdin.writable) {
      try {
        startBridge();
      } catch (error) {
        reject(error);
        return;
      }
    }
    if (!python || !python.stdin.writable) {
      reject(new Error("Python bridge is not available"));
      return;
    }
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    try {
      python.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
    } catch (error) {
      pending.delete(id);
      reject(error);
      return;
    }
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error(`${method} timed out`));
      }
    }, method === "run_pipeline" || method === "run_autopilot" || method === "material_eval_suite" ? 300000 : 30000);
  });
}

function createWindow() {
  const windowOpts = {
    width: 1520,
    height: 960,
    minWidth: 1180,
    minHeight: 720,
    backgroundColor: "#080b10",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  };
  // titleBarOverlay is unstable on some Linux Electron builds — keep native frame there.
  if (process.platform === "darwin" || process.platform === "win32") {
    windowOpts.titleBarStyle = "hidden";
    windowOpts.titleBarOverlay = { color: "#080b10", symbolColor: "#9ca7b4", height: 38 };
  }

  mainWindow = new BrowserWindow(windowOpts);
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.webContents.on("did-fail-load", (_e, code, desc, url) => {
    console.error("did-fail-load", code, desc, url);
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else if (!app.isPackaged) {
    mainWindow.loadURL("http://127.0.0.1:5173");
  } else {
    mainWindow.loadFile(path.join(app.getAppPath(), "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  console.log(`LatticeZero userData=${userDataPath} dataDir=${resolveDataDir()}`);
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
