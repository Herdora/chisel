# Keys & Caches Examples

This directory contains PyTorch model examples that demonstrate GPU profiling with the `capture_model` decorator. Each example can run locally for testing or on cloud GPUs with automatic profiling.

## How to Run

From the `examples/` directory:

```bash
# Test locally (if you have a GPU)
python basic_models/simple_cnn.py

# Run on cloud GPU with profiling
kandc run python basic_models/simple_cnn.py

# Or run locally with profiling and upload
kandc capture python basic_models/simple_cnn.py

## Examples

### Basic Models
**`basic_models/linear_regression.py`** - Linear regression and MLP
```bash
kandc run python basic_models/linear_regression.py
```

**`basic_models/simple_cnn.py`** - Basic CNN for image classification
```bash
kandc run python basic_models/simple_cnn.py
```

**`basic_models/instance_wrapper_example.py`** - `capture_model_instance` demo
```bash
kandc run python basic_models/instance_wrapper_example.py
```

### NLP Models
**`nlp_models/transformer_example.py`** - Custom transformer implementation
```bash
kandc run python nlp_models/transformer_example.py
```

**`nlp_models/pretrained_models.py`** - HuggingFace DistilBERT integration
```bash
kandc run --requirements requirements_examples/nlp_requirements.txt -- python nlp_models/pretrained_models.py
```

**`nlp_models/huggingface_download_example.py`** - Downloads real HF model weights
```bash
kandc run --requirements requirements_examples/nlp_requirements.txt -- python nlp_models/huggingface_download_example.py
```

### Vision Models
**`vision_models/resnet_example.py`** - ResNet-18 and EfficientNet architectures
```bash
kandc run python vision_models/resnet_example.py
```

### Vision-Language Models
**`vlm_models/clip_example.py`** - OpenAI CLIP for image-text understanding
```bash
kandc run --requirements requirements_examples/vlm_requirements.txt -- python vlm_models/clip_example.py
```

### Generative Models
**`generative_models/gan_example.py`** - DCGAN and VAE implementations
```bash
kandc run python generative_models/gan_example.py
```

### Edge Cases & Testing
**`edge_cases/basic_model_call.py`** - Simplest model usage
```bash
kandc run python edge_cases/basic_model_call.py
```

**`edge_cases/model_with_args.py`** - Command line arguments demo
```bash
# Interactive format
kandc run python edge_cases/model_with_args.py --model-size large --batch-size 16

# Separator format
kandc run --app-name "args-test" --gpu 2 -- python edge_cases/model_with_args.py --model-size large --batch-size 16
```

**`edge_cases/long_running_demo.py`** - Long-running demo (~3 mins) with real-time streaming
```bash
# Quick demo (~1 minute)
kandc run python edge_cases/long_running_demo.py --epochs 3 --num-batches 8

# Full demo (~3 minutes)
kandc run python edge_cases/long_running_demo.py --epochs 8 --validate --verbose

# With custom config
kandc run --app-name "streaming-demo" --gpu 2 -- python edge_cases/long_running_demo.py --epochs 10 --validate
```

**`edge_cases/large_file_test.py`** - Large file caching test
```bash
kandc run python edge_cases/large_file_test.py
```

### Nested Directory Tests
**`nested_directory/parent_dir_test.py`** - Parent directory upload test
```bash
kandc run --upload-dir examples -- python nested_directory/parent_dir_test.py
```

**`nested_directory/deep/deeper/deepest_model.py`** - Deeply nested model
```bash
kandc run python nested_directory/deep/deeper/deepest_model.py
```

### Requirements Examples
**`requirements_examples/model_with_custom_requirements.py`** - Custom requirements demo
```bash
kandc run --requirements requirements_examples/scientific_requirements.txt -- python requirements_examples/model_with_custom_requirements.py
```

### C++ Integration
**`cpp_gpu_check/`** - C++ GPU availability checker with Python `ctypes`
```bash
cd cpp_gpu_check && make
python3 check_gpu.py
kandc capture -- python3 check_gpu.py
```
