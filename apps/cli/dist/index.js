"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
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

// src/index.ts
var import_commander = require("commander");

// src/commands/configure.ts
var import_inquirer = __toESM(require("inquirer"));
var import_chalk = __toESM(require("chalk"));
var import_ora = __toESM(require("ora"));

// src/services/config.service.ts
var import_fs = require("fs");
var path = __toESM(require("path"));
var os = __toESM(require("os"));
var ConfigServiceImpl = class {
  configPath;
  constructor() {
    this.configPath = path.join(os.homedir(), ".chisel");
  }
  getConfigPath() {
    return this.configPath;
  }
  async load() {
    try {
      const data = await import_fs.promises.readFile(this.configPath, "utf-8");
      return JSON.parse(data);
    } catch (error) {
      if (error.code === "ENOENT") {
        return null;
      }
      throw new Error(`Failed to load config: ${error.message}`);
    }
  }
  async save(config) {
    try {
      const data = JSON.stringify(config, null, 2);
      await import_fs.promises.writeFile(this.configPath, data, {
        encoding: "utf-8",
        mode: 384
        // Read/write for owner only
      });
    } catch (error) {
      throw new Error(`Failed to save config: ${error.message}`);
    }
  }
};
var configService = new ConfigServiceImpl();

// src/services/digitalocean.service.ts
var import_dots_wrapper = require("dots-wrapper");
var DigitalOceanService = class _DigitalOceanService {
  apiClient;
  constructor(apiKey) {
    this.apiClient = (0, import_dots_wrapper.createApiClient)({ token: apiKey });
  }
  async validateApiKey() {
    try {
      const response = await this.apiClient.account.getAccount();
      if (response.data?.account) {
        return {
          valid: true,
          account: response.data.account
        };
      } else {
        return {
          valid: false,
          error: "Invalid response format from Digital Ocean API"
        };
      }
    } catch (error) {
      let errorMessage = "Unknown error occurred";
      if (error.response?.status === 401) {
        errorMessage = "Invalid API key - authentication failed";
      } else if (error.response?.status === 403) {
        errorMessage = "API key does not have required permissions";
      } else if (error.response?.status === 429) {
        errorMessage = "Rate limit exceeded - please try again later";
      } else if (error.code === "ENOTFOUND" || error.code === "ECONNREFUSED") {
        errorMessage = "Network error - check your internet connection";
      } else if (error.message) {
        errorMessage = error.message;
      }
      return {
        valid: false,
        error: errorMessage
      };
    }
  }
  static async createFromConfig() {
    const config = await configService.load();
    if (!config?.digitalOcean?.apiKey) {
      return null;
    }
    return new _DigitalOceanService(config.digitalOcean.apiKey);
  }
};
var digitalOceanService = {
  create: (apiKey) => new DigitalOceanService(apiKey),
  createFromConfig: () => DigitalOceanService.createFromConfig()
};

// src/commands/configure.ts
async function configureCommand(options) {
  if (options.show) {
    await showConfiguration();
  } else {
    await interactiveConfigure();
  }
}
async function showConfiguration() {
  try {
    const config = await configService.load();
    if (!config) {
      console.log(import_chalk.default.yellow("No configuration found. Run `chisel configure` to set up."));
      return;
    }
    console.log(import_chalk.default.bold("\nChisel Configuration:"));
    console.log(import_chalk.default.gray("\u2500".repeat(40)));
    console.log(`Version: ${config.version}`);
    if (config.digitalOcean?.apiKey) {
      const maskedKey = maskApiKey(config.digitalOcean.apiKey);
      console.log(`Digital Ocean API Key: ${maskedKey}`);
    }
    console.log(import_chalk.default.gray("\u2500".repeat(40)));
    console.log(import_chalk.default.gray(`
Config location: ${configService.getConfigPath()}`));
  } catch (error) {
    console.error(import_chalk.default.red("Error loading configuration:"), error.message);
    process.exit(1);
  }
}
async function interactiveConfigure() {
  console.log(import_chalk.default.bold("\nChisel Configuration Setup"));
  console.log(import_chalk.default.gray("\u2500".repeat(50)));
  console.log(import_chalk.default.gray("This will configure your Digital Ocean API key for chisel."));
  console.log(import_chalk.default.gray("Your API key will be validated and stored securely.\n"));
  let existingConfig = null;
  try {
    existingConfig = await configService.load();
  } catch (error) {
  }
  if (existingConfig?.digitalOcean?.apiKey) {
    const maskedKey = maskApiKey(existingConfig.digitalOcean.apiKey);
    console.log(import_chalk.default.green(`\u2713 API key already configured: ${maskedKey}`));
    const { shouldUpdate } = await import_inquirer.default.prompt([
      {
        type: "confirm",
        name: "shouldUpdate",
        message: "Would you like to update your API key?",
        default: false
      }
    ]);
    if (!shouldUpdate) {
      console.log(import_chalk.default.blue("\nKeeping existing configuration."));
      return;
    }
    console.log(import_chalk.default.gray("\nUpdating your API key...\n"));
  } else {
    console.log(import_chalk.default.blue("\u2139 First time setup detected"));
    console.log(import_chalk.default.gray("You can get your API key from: https://cloud.digitalocean.com/account/api/tokens\n"));
  }
  const answers = await import_inquirer.default.prompt([
    {
      type: "password",
      name: "doApiKey",
      message: "Digital Ocean API key (dop_v1_...):",
      mask: "*",
      validate: (input) => {
        if (!input || input.trim() === "") {
          return "API key is required";
        }
        const trimmed = input.trim();
        if (!isValidDigitalOceanApiKey(trimmed)) {
          return "Invalid Digital Ocean API key format. Expected format: dop_v1_...";
        }
        return true;
      }
    }
  ]);
  const apiKey = answers.doApiKey.trim();
  const spinner = (0, import_ora.default)("Validating API key with Digital Ocean...").start();
  const doService = digitalOceanService.create(apiKey);
  const validationResult = await doService.validateApiKey();
  if (!validationResult.valid) {
    spinner.fail("API key validation failed");
    console.error(import_chalk.default.red(`
${validationResult.error}`));
    console.error(import_chalk.default.gray("\nPlease check your API key and try again."));
    console.error(import_chalk.default.gray("You can generate a new API key at: https://cloud.digitalocean.com/account/api/tokens"));
    process.exit(1);
  }
  spinner.succeed("API key validated successfully!");
  if (validationResult.account) {
    console.log(import_chalk.default.gray(`Account: ${validationResult.account.email}`));
    console.log(import_chalk.default.gray(`Status: ${validationResult.account.status}`));
    if (validationResult.account.email_verified) {
      console.log(import_chalk.default.green("\u2713 Email verified"));
    } else {
      console.log(import_chalk.default.yellow("\u26A0 Email not verified"));
    }
  }
  const config = {
    version: "1.0",
    digitalOcean: {
      apiKey
    }
  };
  try {
    await configService.save(config);
    console.log(import_chalk.default.green("\n\u2713 Configuration saved successfully!"));
    console.log(import_chalk.default.gray(`Config location: ${configService.getConfigPath()}`));
  } catch (error) {
    console.error(import_chalk.default.red("\nError saving configuration:"), error.message);
    process.exit(1);
  }
}
function maskApiKey(apiKey) {
  if (apiKey.length <= 8) {
    return "****";
  }
  return `${apiKey.substring(0, 4)}...${apiKey.substring(apiKey.length - 4)}`;
}
function isValidDigitalOceanApiKey(apiKey) {
  const doApiKeyPattern = /^dop_v1_[a-f0-9]{64}$/i;
  return doApiKeyPattern.test(apiKey);
}

// src/commands/up.ts
var import_execa = require("execa");
var import_chalk2 = __toESM(require("chalk"));
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
  console.log(import_chalk2.default.blue("\u{1F680} Bootstrapping Chisel orchestrator locally (with web UI)..."));
  console.log(import_chalk2.default.gray("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"));
  console.log(import_chalk2.default.yellow("\u{1F4E6} Installing dependencies for orchestrator and UI..."));
  await (0, import_execa.execa)("npm", ["install"], { cwd: "./apps/orchestrator" });
  await (0, import_execa.execa)("npm", ["install"], { cwd: "./apps/orchestrator/ui" });
  console.log(import_chalk2.default.green("\u2705 Dependencies installed successfully"));
  console.log(import_chalk2.default.yellow("\u{1F528} Building orchestrator and UI..."));
  await (0, import_execa.execa)("npm", ["run", "build"], { cwd: "./apps/orchestrator/ui" });
  await (0, import_execa.execa)("npm", ["run", "build"], { cwd: "./apps/orchestrator" });
  console.log(import_chalk2.default.green("\u2705 Build completed successfully"));
  console.log(import_chalk2.default.yellow("\u{1F680} Starting orchestrator backend and web UI (dev mode)..."));
  console.log(import_chalk2.default.gray("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"));
  const orchestratorUrl = "http://localhost:3001";
  const uiUrl = "http://localhost:5173";
  console.log(import_chalk2.default.cyan.bold("\u{1F3AF} Orchestrator backend: ") + import_chalk2.default.underline(orchestratorUrl));
  console.log(import_chalk2.default.cyan.bold("\u{1F3AF} Web UI: ") + import_chalk2.default.underline(uiUrl));
  console.log(import_chalk2.default.gray("\u{1F4A1} Press Ctrl+C to stop both servers"));
  console.log(import_chalk2.default.gray("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"));
  await (0, import_execa.execa)("npx", [
    "concurrently",
    '"npm run dev"',
    '"cd ui && npm run dev"'
  ], {
    cwd: "./apps/orchestrator",
    stdio: "inherit"
  });
}

// src/index.ts
var program = new import_commander.Command();
program.name("chisel").description("Chisel CLI for GPU profiling").version("0.1.0");
program.command("configure").description("Configure chisel settings (Digital Ocean API key)").option("--show", "Display current configuration with masked sensitive values").action(async (options) => {
  await configureCommand(options);
});
program.command("ncu").option("--metrics <metrics>", "NVIDIA metrics to collect").option("--script <path>", "Python script to run").action((options) => {
  console.log("Running NCU with options:", options);
});
program.command("torchprof").option("--trace-output <file>").option("--script <path>").action((options) => {
  console.log("Running torchprof with options:", options);
});
program.command("up").description("Bootstrap the orchestrator (local or DigitalOcean)").option("--local", "Run orchestrator locally (default)", true).option("--cloud", "Deploy to DigitalOcean cloud").action(async (options) => {
  if (options.cloud) {
    await bootstrapOrchestrator();
  } else {
    await bootstrapOrchestratorLocal();
  }
});
program.parse();
