"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/commands/up.ts
var up_exports = {};
__export(up_exports, {
  bootstrapOrchestrator: () => bootstrapOrchestrator,
  bootstrapOrchestratorLocal: () => bootstrapOrchestratorLocal
});
module.exports = __toCommonJS(up_exports);
var import_execa = require("execa");
var import_chalk = __toESM(require("chalk"));
async function getSshKeyId() {
  const { stdout } = await (0, import_execa.execa)("doctl", ["compute", "ssh-key", "list", "--format", "ID,Name"]);
  const lines = stdout.trim().split("\n").slice(1);
  if (lines.length === 0) {
    throw new Error("No SSH keys found in DigitalOcean. Add one via `doctl compute ssh-key create` or the dashboard.");
  }
  const firstKeyId = lines[0].split(/\s+/)[0];
  console.log(`Using SSH key: ${lines[0]}`);
  return firstKeyId;
}
async function bootstrapOrchestrator() {
  console.log("Bootstrapping orchestrator on DigitalOcean...");
  const dropletName = "chisel-orch-001";
  const sshKeyId = await getSshKeyId();
  const { stdout: createOutput } = await (0, import_execa.execa)("doctl", [
    "compute",
    "droplet",
    "create",
    dropletName,
    "--region",
    "nyc3",
    "--image",
    "ubuntu-22-04-x64",
    "--size",
    "s-2vcpu-4gb",
    "--ssh-keys",
    sshKeyId,
    "--wait"
  ]);
  console.log("Droplet created:", createOutput);
  const { stdout: ipOutput } = await (0, import_execa.execa)("doctl", ["compute", "droplet", "get", dropletName, "--format", "PublicIPv4"]);
  const ip = ipOutput.trim().split("\n")[1];
  console.log("Droplet IP:", ip);
  await (0, import_execa.execa)("tar", ["czf", "chisel.tar.gz", "."]);
  await (0, import_execa.execa)("scp", ["chisel.tar.gz", `root@${ip}:/root/`]);
  const sshCommands = `
    console.log('Installing Node.js...');
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -;
    apt-get install -y nodejs;
    tar xzf chisel.tar.gz -C /root/chisel/;
    cd /root/chisel/apps/orchestrator;
    npm install --omit=dev;
    npm run build;
    nohup node dist/index.js > out.log 2>&1 &;
  `;
  await (0, import_execa.execa)("ssh", [`root@${ip}`, sshCommands]);
  console.log(`\u2713 Orchestrator online at http://${ip}:3000`);
}
async function bootstrapOrchestratorLocal() {
  console.log(import_chalk.default.blue("\u{1F680} Bootstrapping Chisel orchestrator locally..."));
  console.log(import_chalk.default.gray("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"));
  console.log(import_chalk.default.yellow("\u{1F4E6} Installing dependencies..."));
  await (0, import_execa.execa)("npm", ["install"], { cwd: "./apps/orchestrator" });
  console.log(import_chalk.default.green("\u2705 Dependencies installed successfully"));
  console.log(import_chalk.default.yellow("\u{1F528} Building orchestrator..."));
  await (0, import_execa.execa)("npm", ["run", "build"], { cwd: "./apps/orchestrator" });
  console.log(import_chalk.default.green("\u2705 Build completed successfully"));
  console.log(import_chalk.default.yellow("\u{1F680} Starting orchestrator..."));
  console.log(import_chalk.default.gray("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"));
  const orchestratorUrl = "http://localhost:3001";
  console.log(import_chalk.default.cyan.bold("\u{1F3AF} Orchestrator is starting up..."));
  console.log(import_chalk.default.cyan(`\u{1F4E1} Server will be available at: ${import_chalk.default.underline(orchestratorUrl)}`));
  console.log(import_chalk.default.gray("\u{1F4A1} Press Ctrl+C to stop the server"));
  console.log(import_chalk.default.gray("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"));
  await (0, import_execa.execa)("node", ["dist/index.js"], {
    cwd: "./apps/orchestrator",
    stdio: "inherit"
    // Show output in real-time
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  bootstrapOrchestrator,
  bootstrapOrchestratorLocal
});
