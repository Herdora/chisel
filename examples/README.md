# Chisel Examples

This directory contains comprehensive examples demonstrating the `capture_model` decorator from Chisel CLI. These examples show how to profile PyTorch models on cloud GPUs with automatic tracing and performance analysis.

## 🚀 Quick Start

```bash
# Install Chisel
pip install git+https://github.com/Herdora/chisel.git@dev

# Run any example locally first
python basic_models/simple_cnn.py

# Run on cloud GPU with profiling
chisel python basic_models/simple_cnn.py
```

## 📁 Directory Structure

```
examples/
├── README.md                           # This file
├── requirements.txt                    # Main requirements file
├── test_model_profiling.py            # Original test file
│
├── basic_models/                       # Simple model examples
│   ├── simple_cnn.py                  # Basic CNN for image classification
│   └── linear_regression.py           # Linear models and MLP
│
├── nlp_models/                         # Natural Language Processing
│   ├── transformer_example.py         # Custom transformer implementation
│   ├── pretrained_models.py           # HuggingFace model integration
│   └── huggingface_download_example.py # Download real HF weights + profiling
│
├── vision_models/                      # Computer Vision
│   └── resnet_example.py              # ResNet-18 and EfficientNet-style models
│
├── generative_models/                  # Generative Models
│   └── gan_example.py                 # GAN and VAE implementations
│
├── edge_cases/                         # Testing edge cases
│   ├── basic_model_call.py            # Simplest possible model call
│   ├── model_with_args.py             # Command line arguments handling
│   ├── long_running_demo.py           # Long-running demo with real-time streaming (~3 mins)
│   └── large_file_test.py             # Large file caching test
│
├── nested_directory/                   # Directory structure tests
│   ├── parent_dir_test.py             # Parent directory upload test
│   └── deep/deeper/deepest_model.py   # Deeply nested model
│
├── requirements_examples/              # Custom requirements
│   ├── minimal_requirements.txt       # Minimal PyTorch setup
│   ├── nlp_requirements.txt          # NLP-specific packages
│   ├── vision_requirements.txt       # Computer vision packages
│   ├── scientific_requirements.txt   # Scientific computing
│   └── model_with_custom_requirements.py # Custom requirements demo
│
└── requirements/                       # Additional requirements
    └── dev.txt                        # Development requirements
```

## 🎯 Core Concept: `capture_model` Decorator

All examples use the `capture_model` decorator to automatically profile PyTorch models:

```python
from chisel import capture_model

@capture_model(model_name="MyModel", record_shapes=True, profile_memory=True)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
```

**Key Features:**
- **Automatic profiling**: Every `forward()` call is profiled when running on Chisel backend
- **Local pass-through**: No overhead when running locally (`python script.py`)
- **Cloud activation**: Profiling activates automatically with `chisel python script.py`
- **Detailed traces**: Chrome trace format with layer-level timing and shape information

## 📚 Example Categories

### 1. Basic Models (`basic_models/`)

**Simple CNN** (`simple_cnn.py`)
```bash
# Test locally
python basic_models/simple_cnn.py

# Run on cloud GPU
chisel python basic_models/simple_cnn.py
```
- Basic convolutional neural network
- Image classification architecture
- Demonstrates conv layers, pooling, and fully connected layers

**Linear Models** (`linear_regression.py`)
```bash
chisel python basic_models/linear_regression.py
```
- Linear regression and multi-layer perceptron
- Synthetic data generation
- Comparison between simple and complex models

### 2. NLP Models (`nlp_models/`)

**Custom Transformer** (`transformer_example.py`)
```bash
chisel python nlp_models/transformer_example.py
```
- GPT-like and BERT-like transformer implementations
- Attention mechanisms and positional encoding
- Causal and bidirectional attention patterns

**Pretrained Models** (`pretrained_models.py`)
```bash
# Requires transformers library
chisel python nlp_models/pretrained_models.py --requirements requirements_examples/nlp_requirements.txt
```
- HuggingFace DistilBERT integration
- Wrapped pretrained models with profiling
- Text classification examples

**HuggingFace Download Example** (`huggingface_download_example.py`)
```bash
# Downloads actual model weights from HuggingFace
chisel python nlp_models/huggingface_download_example.py --requirements requirements_examples/nlp_requirements.txt
```
- **Downloads real model weights** from HuggingFace Hub
- DistilBERT and GPT-2 examples with actual pretrained weights
- Automatic model caching for faster subsequent runs
- Demonstrates both encoder and decoder architectures

### 3. Vision Models (`vision_models/`)

**ResNet Example** (`resnet_example.py`)
```bash
chisel python vision_models/resnet_example.py
```
- ResNet-18 implementation with residual blocks
- EfficientNet-style mobile architecture
- Multiple input size testing

### 4. Generative Models (`generative_models/`)

**GAN Example** (`gan_example.py`)
```bash
chisel python generative_models/gan_example.py
```
- DCGAN-style Generator and Discriminator
- Variational Autoencoder (VAE) with encoder/decoder
- Reparameterization trick demonstration

## 🧪 Edge Cases (`edge_cases/`)

### Basic Model Call
```bash
chisel python edge_cases/basic_model_call.py
```
Tests the simplest possible `capture_model` usage.

### Command Line Arguments

Chisel CLI supports **two clean argument formats** to handle chisel configuration and script arguments:

```bash
# 1. Separator format - chisel flags first, then -- separator (RECOMMENDED)
chisel --app-name "args-test" --gpu 2 -- python edge_cases/model_with_args.py --model-size large --batch-size 16

# 2. Interactive format - script args only (prompts for chisel config)
chisel python edge_cases/model_with_args.py --model-size large --batch-size 16 --num-layers 5
```

**Key Features:**
- **Clean Separation**: Use `--` for explicit separation between chisel and script arguments
- **Interactive Fallback**: Script args only triggers interactive mode for chisel configuration
- **Error Prevention**: Mixing chisel flags with script args is not allowed - keeps things simple
- **Helpful Errors**: Clear messages guide you to the correct format

### Long-Running Demo with Real-Time Streaming

```bash
# Quick demo (~1 minute)
chisel python edge_cases/long_running_demo.py --epochs 3 --num-batches 8

# Full demo (~3 minutes with validation)
chisel python edge_cases/long_running_demo.py --epochs 8 --validate --verbose

# Separator format with custom configuration
chisel --app-name "streaming-demo" --gpu 2 -- python edge_cases/long_running_demo.py --epochs 10 --model-size large --validate
```

**Perfect for testing:**
- **Real-time stdout streaming** - See logs as they happen
- **Progress tracking** - Visual progress bars and status updates
- **Comprehensive logging** - Detailed training metrics and timestamps
- **Flexible duration** - Configure epochs and batches for desired runtime
- **Model profiling** - Uses `@capture_model` decorator for GPU profiling
 
### Large File Handling
```bash
chisel python edge_cases/large_file_test.py
```
- Creates a 100MB test file
- Tests Chisel's large file caching system
- Demonstrates file upload optimization

**⚠️ Important:** When uploading large files, **do not refresh the browser** during upload. This can interrupt the caching process.

### HuggingFace Model Downloads
```bash
# Downloads real model weights (DistilBERT ~250MB, GPT-2 ~500MB)
chisel python nlp_models/huggingface_download_example.py --requirements requirements_examples/nlp_requirements.txt
```
- **Downloads actual pretrained weights** from HuggingFace Hub
- Models cached locally in `model_cache/` directory
- First run downloads weights, subsequent runs use cache
- Demonstrates profiling real, production-ready models

## 📂 Directory Structure Tests (`nested_directory/`)

### Parent Directory Upload
```bash
# From chisel root directory
chisel python examples/nested_directory/parent_dir_test.py --upload-dir examples

# Test uploading parent directory
cd examples/nested_directory
chisel python parent_dir_test.py --upload-dir ../
```

### Deeply Nested Model
```bash
# From examples directory
chisel python nested_directory/deep/deeper/deepest_model.py
```
Tests running models from deeply nested directory structures.

## 📦 Requirements Examples

### Using Custom Requirements
```bash
# Minimal setup
chisel python basic_models/simple_cnn.py --requirements requirements_examples/minimal_requirements.txt

# NLP-specific packages
chisel python nlp_models/transformer_example.py --requirements requirements_examples/nlp_requirements.txt

# Vision packages
chisel python vision_models/resnet_example.py --requirements requirements_examples/vision_requirements.txt

# Scientific computing
chisel python requirements_examples/model_with_custom_requirements.py --requirements requirements_examples/scientific_requirements.txt

# Separator format - chisel flags first, then -- separator
chisel --requirements requirements_examples/minimal_requirements.txt --gpu 2 -- python edge_cases/model_with_args.py --model-size large --epochs 5
```

### Requirements Files Available:
- `minimal_requirements.txt` - Just PyTorch and NumPy
- `nlp_requirements.txt` - Transformers, tokenizers, datasets
- `vision_requirements.txt` - Computer vision libraries
- `scientific_requirements.txt` - Scientific computing stack

## 🎮 Interactive Examples

All examples support **two clean argument formats**:

```bash
# 1. Interactive format - script args only (prompts for chisel config)
chisel python basic_models/simple_cnn.py
chisel python edge_cases/model_with_args.py --model-size large --epochs 10 --batch-size 32
# CLI will prompt for: app name, upload directory, requirements file, GPU configuration

# 2. Separator format - chisel flags first, then -- separator
chisel --app-name "configurable-test" --gpu 2 -- \
  python edge_cases/model_with_args.py \
  --model-size large \
  --epochs 10 \
  --batch-size 32

# Traditional chisel-only configuration (still supported)
chisel python basic_models/simple_cnn.py \
  --app-name "cnn-test" \
  --upload-dir "." \
  --requirements "requirements.txt" \
  --gpu 2
```

## 🔧 GPU Configuration

Choose GPU count based on your model size:

| GPUs | Memory | Use Case                       | Example   |
| ---- | ------ | ------------------------------ | --------- |
| 1    | 80GB   | Development, small models      | `--gpu 1` |
| 2    | 160GB  | Medium models, faster training | `--gpu 2` |
| 4    | 320GB  | Large models, high throughput  | `--gpu 4` |
| 8    | 640GB  | Massive models, research scale | `--gpu 8` |

## 📊 Model Profiling Features

The `capture_model` decorator provides:

### Automatic Tracing
- **Layer-level timing**: See which layers are bottlenecks
- **Memory profiling**: Track GPU memory usage
- **Shape recording**: Debug tensor dimension issues
- **Chrome trace format**: Visualize in chrome://tracing

### Multiple Forward Passes
Each forward pass is automatically numbered and traced:
```
🔍 [capture_model] Tracing MyModel forward pass #1
💾 [capture_model] Saved trace: MyModel_forward_001.json
🔍 [capture_model] Tracing MyModel forward pass #2
💾 [capture_model] Saved trace: MyModel_forward_002.json
```

### Layer Analysis
Automatic layer-level analysis with timing and shape information:
```
📊 Layer Analysis for MyModel
─────────────────────────────────────────────────────────────
Layer                     Calls    Total Time (ms)  Avg Time (ms)   Shapes
─────────────────────────────────────────────────────────────
Unknown_addmm             1        45.23           45.23           (8, 512)
Unknown_relu               1        12.45           12.45           (8, 512)
Unknown_dropout            1        8.91            8.91            (8, 512)
─────────────────────────────────────────────────────────────
TOTAL                              66.59
```

## 🚀 Running Examples

### Local Testing (No Profiling)
```bash
# Test any example locally first
python basic_models/simple_cnn.py
python nlp_models/transformer_example.py
python vision_models/resnet_example.py
```

### Cloud GPU Execution (With Profiling)
```bash
# Run with automatic profiling on cloud GPUs
chisel python basic_models/simple_cnn.py
chisel python nlp_models/transformer_example.py
chisel python vision_models/resnet_example.py
```

### Advanced Usage

**Separator Format (RECOMMENDED)**
```bash
# Clean separation with -- for complex configurations
chisel --app-name "complex-training" --gpu 4 --requirements requirements_examples/scientific_requirements.txt -- \
  python edge_cases/model_with_args.py \
  --model-size xl \
  --num-layers 8 \
  --batch-size 64 \
  --epochs 100 \
  --verbose
```

**Interactive Format**
```bash
# Script args only - prompts for chisel configuration
chisel python edge_cases/model_with_args.py \
  --model-size xl \
  --num-layers 8 \
  --batch-size 64 \
  --epochs 100 \
  --verbose
```

**Traditional Chisel-Only**
```bash
# Traditional approach (still supported)
chisel python generative_models/gan_example.py \
  --app-name "gan-training" \
  --gpu 4 \
  --requirements requirements_examples/vision_requirements.txt

# From nested directory with parent upload
cd nested_directory
chisel python parent_dir_test.py \
  --app-name "nested-test" \
  --upload-dir ../ \
  --gpu 1
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure chisel is installed
pip install git+https://github.com/Herdora/chisel.git@dev

# Check if torch is available
python -c "import torch; print(torch.__version__)"
```

**Missing Dependencies:**
```bash
# Install specific requirements
pip install -r requirements_examples/nlp_requirements.txt
pip install -r requirements_examples/vision_requirements.txt
```

**File Not Found:**
```bash
# Make sure you're in the examples directory
cd examples
ls basic_models/  # Should show simple_cnn.py, etc.
```

**Large File Upload Issues:**
- **Do not refresh browser** during large file uploads
- First upload may take 2-5 minutes for 1GB files
- Subsequent uploads are much faster (cached)
- Check `.chiselignore` to exclude unnecessary large files

### Authentication Issues
```bash
# Clear credentials and re-authenticate
chisel --logout
chisel python basic_models/simple_cnn.py  # Will prompt for login
```

## 💡 Best Practices

### 1. Test Locally First
```bash
# Always test locally before cloud execution
python your_model.py
chisel python your_model.py
```

### 2. Use Appropriate GPU Count
- Start with 1 GPU for development
- Scale up based on model size and memory needs
- Use 4-8 GPUs for large models or high throughput

### 3. Optimize Upload Size
- Use `.chiselignore` to exclude unnecessary files
- Upload only the `src/` directory if possible
- Consider custom requirements files

### 4. Monitor Resource Usage
- Enable memory profiling: `profile_memory=True`
- Use shape recording for debugging: `record_shapes=True`
- Check trace files for performance bottlenecks

### 5. Handle Dependencies Gracefully
```python
try:
    import optional_package
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False

# Use HAS_OPTIONAL to conditionally enable features
```

## 📈 Performance Tips

1. **Batch Size**: Use larger batches for better GPU utilization
2. **Memory Management**: Clear GPU cache with `torch.cuda.empty_cache()`
3. **Mixed Precision**: Use `torch.cuda.amp` for faster training
4. **Data Loading**: Minimize CPU-GPU transfers
5. **Model Compilation**: Consider `torch.compile()` for newer PyTorch versions

## 🔗 Related Resources

- **Main Documentation**: `../docs/`
- **Chisel CLI Source**: `../src/chisel/`
- **Issue Tracker**: [GitHub Issues](https://github.com/Herdora/chisel/issues)

## 🆘 Getting Help

1. **Check examples**: Look for similar use cases in this directory
2. **Run locally first**: Debug issues without cloud overhead
3. **Check requirements**: Ensure all dependencies are installed
4. **File issues**: Report bugs on GitHub with example code

---

## 📝 Example Usage Summary

**Two Clean Argument Formats:**

```bash
# 1. Interactive format - script args only (prompts for chisel config)
chisel python basic_models/simple_cnn.py
chisel python edge_cases/model_with_args.py --model-size large --batch-size 16
chisel python edge_cases/long_running_demo.py --epochs 5 --validate

# 2. Separator format - chisel flags first, then -- separator
chisel --app-name "test" --gpu 2 -- python edge_cases/model_with_args.py --model-size large --batch-size 16
chisel --app-name "streaming-demo" --gpu 2 -- python edge_cases/long_running_demo.py --epochs 8 --validate --verbose
chisel --gpu 4 --requirements requirements_examples/nlp_requirements.txt -- python nlp_models/pretrained_models.py

# Traditional chisel-only configuration (still supported)
chisel python generative_models/gan_example.py \
  --app-name "gan-experiment" \
  --upload-dir "." \
  --requirements "requirements.txt" \
  --gpu 2
```

**Happy profiling with Chisel! 🚀**
