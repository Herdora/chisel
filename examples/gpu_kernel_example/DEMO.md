## Run locally
```bash
kandc capture -- python gpu_kernel_example/run_kernel.py
```


```bash
kandc sweep capture \
  --app-name sd-sweep \
  --configs-dir diffusion_configs \
  --script diffusion_models/text2img_inference.py \
  --gpus 0,1,2,3,4,5,6,7 --per-run-gpus 1 \
  --code-snapshot-dir . \
  --auto-confirm \
  --tmux
```


```bash
kandc capture -- python basic_model/simple_cnn.py
```

