const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("lattice", {
  request: (method, params) => ipcRenderer.invoke("cadflow:request", method, params),
  chooseDirectory: () => ipcRenderer.invoke("dialog:directory"),
  reveal: (filePath) => ipcRenderer.invoke("shell:reveal", filePath),
  open: (target) => ipcRenderer.invoke("shell:open", target),
  onEvent: (callback) => {
    const handler = (_, event) => callback(event);
    ipcRenderer.on("cadflow:event", handler);
    return () => ipcRenderer.removeListener("cadflow:event", handler);
  },
  platform: process.platform,
});
