# Chisel Examples

This directory contains PyTorch model examples that demonstrate GPU profiling with the `capture_model` decorator. Each example can run locally for testing or on cloud GPUs with automatic profiling.

## How to Run

From the `examples/` directory:

```bash
# Test locally first
python basic_models/simple_cnn.py

# Run on cloud GPU with profiling
chisel python basic_models/simple_cnn.py

# Run with specific GPU configuration
chisel --gpu H100:4 python basic_models/simple_cnn.py      # 4x H100 GPUs
chisel --gpu A100-80GB:2 python basic_models/simple_cnn.py # 2x A100-80GB GPUs
chisel --gpu L4:1 python basic_models/simple_cnn.py        # 1x L4 GPU
```

## GPU Options

| GPU Type  | Example Flags   | Memory    | Best For                |
| --------- | --------------- | --------- | ----------------------- |
| A100-40GB | `A100:1-8`      | 40GB each | Cost-effective training |
| A100-80GB | `A100-80GB:1-8` | 80GB each | High-memory models      |
| H100      | `H100:1-8`      | 80GB each | Latest architecture     |
| L4        | `L4:1-8`        | 24GB each | Efficient inference     |

## Examples

### Basic Models
**`basic_models/linear_regression.py`** - Linear regression and MLP
```bash
chisel python basic_models/linear_regression.py
```

**`basic_models/simple_cnn.py`** - Basic CNN for image classification
```bash
chisel python basic_models/simple_cnn.py
```

**`basic_models/instance_wrapper_example.py`** - `capture_model_instance` demo
```bash
chisel python basic_models/instance_wrapper_example.py
```

### NLP Models
**`nlp_models/transformer_example.py`** - Custom transformer implementation
```bash
chisel python nlp_models/transformer_example.py
```

**`nlp_models/pretrained_models.py`** - HuggingFace DistilBERT integration
```bash
chisel --requirements requirements_examples/nlp_requirements.txt -- python nlp_models/pretrained_models.py
```

**`nlp_models/huggingface_download_example.py`** - Downloads real HF model weights
```bash
chisel --requirements requirements_examples/nlp_requirements.txt -- python nlp_models/huggingface_download_example.py
```

### Vision Models
**`vision_models/resnet_example.py`** - ResNet-18 and EfficientNet architectures
```bash
chisel python vision_models/resnet_example.py
```

### Vision-Language Models
**`vlm_models/clip_example.py`** - OpenAI CLIP for image-text understanding
```bash
chisel --requirements requirements_examples/vlm_requirements.txt -- python vlm_models/clip_example.py
```

### Generative Models
**`generative_models/gan_example.py`** - DCGAN and VAE implementations
```bash
chisel python generative_models/gan_example.py
```

### Edge Cases & Testing
**`edge_cases/basic_model_call.py`** - Simplest model usage
```bash
chisel python edge_cases/basic_model_call.py
```

**`edge_cases/model_with_args.py`** - Command line arguments demo
```bash
# Interactive format
chisel python edge_cases/model_with_args.py --model-size large --batch-size 16

# Separator format
chisel --app-name "args-test" --gpu H100:2 -- python edge_cases/model_with_args.py --model-size large --batch-size 16
```

**`edge_cases/long_running_demo.py`** - Long-running demo (~3 mins) with real-time streaming
```bash
# Quick demo (~1 minute)
chisel python edge_cases/long_running_demo.py --epochs 3 --num-batches 8

# Full demo (~3 minutes)
chisel python edge_cases/long_running_demo.py --epochs 8 --validate --verbose

# With custom config
chisel --app-name "streaming-demo" --gpu H100:2 -- python edge_cases/long_running_demo.py --epochs 10 --validate
```

**`edge_cases/large_file_test.py`** - Large file caching test
```bash
chisel python edge_cases/large_file_test.py
```

### Nested Directory Tests
**`nested_directory/parent_dir_test.py`** - Parent directory upload test
```bash
chisel --upload-dir examples -- python nested_directory/parent_dir_test.py
```

**`nested_directory/deep/deeper/deepest_model.py`** - Deeply nested model
```bash
chisel python nested_directory/deep/deeper/deepest_model.py
```

### Requirements Examples
**`requirements_examples/model_with_custom_requirements.py`** - Custom requirements demo
```bash
chisel --requirements requirements_examples/scientific_requirements.txt -- python requirements_examples/model_with_custom_requirements.py
```
