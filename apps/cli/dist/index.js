"use strict";

// src/index.ts
var import_commander = require("commander");
var program = new import_commander.Command();
program.command("ncu").option("--metrics <metrics>", "NVIDIA metrics to collect").option("--script <path>", "Python script to run").action((options) => {
  console.log("Running NCU with options:", options);
});
program.command("torchprof").option("--trace-output <file>").option("--script <path>").action((options) => {
  console.log("Running torchprof with options:", options);
});
program.parse();
