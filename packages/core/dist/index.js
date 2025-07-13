"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
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
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var index_exports = {};
__export(index_exports, {
  GPUModel: () => GPUModel,
  GPUVendor: () => GPUVendor,
  ProfilerType: () => ProfilerType,
  hello: () => hello
});
module.exports = __toCommonJS(index_exports);

// src/enums.ts
var ProfilerType = /* @__PURE__ */ ((ProfilerType2) => {
  ProfilerType2["NCU"] = "ncu";
  ProfilerType2["NSYS"] = "nsys";
  ProfilerType2["TORCHPROF"] = "torchprof";
  ProfilerType2["ROCPROF"] = "rocprof";
  return ProfilerType2;
})(ProfilerType || {});
var GPUVendor = /* @__PURE__ */ ((GPUVendor2) => {
  GPUVendor2["NVIDIA"] = "nvidia";
  GPUVendor2["AMD"] = "amd";
  return GPUVendor2;
})(GPUVendor || {});
var GPUModel = /* @__PURE__ */ ((GPUModel2) => {
  GPUModel2["H100"] = "h100";
  GPUModel2["MI300X"] = "mi300x";
  return GPUModel2;
})(GPUModel || {});

// src/index.ts
var hello = () => "Hello from core";
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  GPUModel,
  GPUVendor,
  ProfilerType,
  hello
});
