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
var import_cors = __toESM(require("@fastify/cors"));
var import_static = __toESM(require("@fastify/static"));
var import_websocket = __toESM(require("@fastify/websocket"));
var import_path = __toESM(require("path"));
console.log(import_chalk.default.blue("\u{1F680} Starting Chisel orchestrator..."));
var server = (0, import_fastify.default)({
  logger: true
});
var jobs = /* @__PURE__ */ new Map();
var setupServer = async () => {
  await server.register(import_cors.default, {
    origin: true
  });
  await server.register(import_websocket.default);
  await server.register(import_static.default, {
    root: import_path.default.join(process.cwd(), "ui/dist"),
    prefix: "/"
  });
  server.get("/api/health", async () => {
    return { status: "ok", timestamp: (/* @__PURE__ */ new Date()).toISOString() };
  });
  server.get("/api/jobs", async () => {
    return Array.from(jobs.values()).sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  });
  server.get("/api/jobs/:id", async (request, reply) => {
    const { id } = request.params;
    const job = jobs.get(id);
    if (!job) {
      return reply.code(404).send({ error: "Job not found" });
    }
    return job;
  });
  server.post("/api/jobs", async (request) => {
    const body = request.body;
    const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const job = {
      id: jobId,
      profiler: body.profiler,
      command: body.command,
      gpu: body.gpu,
      flags: body.flags || {},
      status: "queued",
      createdAt: /* @__PURE__ */ new Date(),
      logs: [`Job ${jobId} created at ${(/* @__PURE__ */ new Date()).toISOString()}`]
    };
    jobs.set(jobId, job);
    setTimeout(() => {
      job.status = "running";
      job.startedAt = /* @__PURE__ */ new Date();
      job.logs.push(`Job ${jobId} started at ${(/* @__PURE__ */ new Date()).toISOString()}`);
      setTimeout(() => {
        job.logs.push(`Running ${job.profiler} profiler on ${job.gpu}...`);
        job.logs.push(`Command: ${job.command}`);
        setTimeout(() => {
          job.status = "completed";
          job.completedAt = /* @__PURE__ */ new Date();
          job.logs.push(`Job ${jobId} completed at ${(/* @__PURE__ */ new Date()).toISOString()}`);
          job.result = {
            metrics: {
              gpuUtilization: Math.random() * 100,
              memoryUsage: Math.random() * 100,
              temperature: 60 + Math.random() * 20
            },
            profileData: {
              duration: Math.random() * 1e3,
              samples: Math.floor(Math.random() * 1e4)
            }
          };
        }, 3e3);
      }, 1e3);
    }, 1e3);
    return { jobId, status: "queued" };
  });
  server.get("/api/jobs/:id/logs", { websocket: true }, (connection, req) => {
    const { id } = req.params;
    const job = jobs.get(id);
    if (!job) {
      connection.socket.send(JSON.stringify({ error: "Job not found" }));
      return;
    }
    connection.socket.send(JSON.stringify({ logs: job.logs }));
    const interval = setInterval(() => {
      const updatedJob = jobs.get(id);
      if (updatedJob && updatedJob.logs.length > job.logs.length) {
        const newLogs = updatedJob.logs.slice(job.logs.length);
        connection.socket.send(JSON.stringify({ logs: newLogs }));
        job.logs = [...updatedJob.logs];
      }
    }, 1e3);
    connection.socket.on("close", () => {
      clearInterval(interval);
    });
  });
  server.get("/*", async (request, reply) => {
    return reply.sendFile("index.html");
  });
};
var start = async () => {
  try {
    console.log(import_chalk.default.yellow("\u{1F527} Attempting to start server..."));
    await setupServer();
    await server.listen({ port: 3001, host: "0.0.0.0" });
    console.log(import_chalk.default.green("\u2705 Orchestrator running on http://localhost:3001"));
    console.log(import_chalk.default.cyan("\u{1F4E1} Ready to accept job requests"));
    console.log(import_chalk.default.blue("\u{1F310} Web UI available at http://localhost:3001"));
    console.log(import_chalk.default.gray("\u{1F4A1} Press Ctrl+C to stop the server"));
  } catch (err) {
    console.error(import_chalk.default.red("\u274C Error starting server:"), err);
    server.log.error(err);
    process.exit(1);
  }
};
start();
