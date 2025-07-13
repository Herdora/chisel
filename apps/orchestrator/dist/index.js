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
var import_fastify = __toESM(require("fastify"));
var import_chalk = __toESM(require("chalk"));
console.log(import_chalk.default.blue("\u{1F680} Starting Chisel orchestrator..."));
var server = (0, import_fastify.default)();
server.get("/", async () => {
  return { hello: "chisel orchestrator" };
});
var start = async () => {
  try {
    console.log(import_chalk.default.yellow("\u{1F527} Attempting to start server..."));
    await server.listen({ port: 3001 });
    console.log(import_chalk.default.green("\u2705 Orchestrator running on http://localhost:3001"));
    console.log(import_chalk.default.cyan("\u{1F4E1} Ready to accept job requests"));
    console.log(import_chalk.default.gray("\u{1F4A1} Press Ctrl+C to stop the server"));
  } catch (err) {
    console.error(import_chalk.default.red("\u274C Error starting server:"), err);
    server.log.error(err);
    process.exit(1);
  }
};
start();
