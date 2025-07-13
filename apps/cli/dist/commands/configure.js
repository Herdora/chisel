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

// src/commands/configure.ts
var configure_exports = {};
__export(configure_exports, {
  configureCommand: () => configureCommand
});
module.exports = __toCommonJS(configure_exports);
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  configureCommand
});
