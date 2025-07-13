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

// src/services/digitalocean.service.ts
var digitalocean_service_exports = {};
__export(digitalocean_service_exports, {
  DigitalOceanService: () => DigitalOceanService,
  digitalOceanService: () => digitalOceanService
});
module.exports = __toCommonJS(digitalocean_service_exports);
var import_dots_wrapper = require("dots-wrapper");

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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  DigitalOceanService,
  digitalOceanService
});
