# 💡 Keys & Caches Examples & Use Cases

Explore real-world examples and learn how to use Keys & Caches for different machine learning workflows. From simple models to complex architectures, these examples will help you master Keys & Caches' capabilities.

## 🎯 Quick Reference

| Category                                       | Use Case                            | Complexity   | GPU Rec. |
| ---------------------------------------------- | ----------------------------------- | ------------ | -------- |
| [Basic Models](#-basic-models)                 | Learning Keys & Caches fundamentals | Beginner     | 1 GPU    |
| [Computer Vision](#-computer-vision)           | Image classification, CNNs          | Intermediate | 1-2 GPUs |
| [NLP & Transformers](#-nlp--transformers)      | Text processing, BERT, GPT          | Advanced     | 2-4 GPUs |
| [Vision-Language](#-vision-language-models)    | CLIP, multi-modal AI                | Advanced     | 2-4 GPUs |
| [Generative Models](#-generative-models)       | GANs, VAEs, diffusion               | Expert       | 4-8 GPUs |
| [Production Workflows](#-production-workflows) | Real-world scenarios                | Expert       | Variable |

---

## 🏗️ Basic Models

Perfect for learning Keys & Caches fundamentals and testing your setup.

### Linear Regression & MLP
```bash
kandc python examples/basic_models/linear_regression.py
```

**What you'll learn:**
- Basic `@capture_model_class` usage
- Synthetic data generation
- Comparing simple vs complex models

**Key Features:**
- Linear regression with polynomial features
- Multi-layer perceptron comparison
- Automatic profiling of forward passes

### Simple CNN
```bash
kandc python examples/basic_models/simple_cnn.py
```

**What you'll learn:**
- Convolutional neural networks
- Image classification architecture
- Conv layers, pooling, and fully connected layers

### Instance Wrapper Demo
```bash
kandc python examples/basic_models/instance_wrapper_example.py
```

**What you'll learn:**
- Using `capture_model_instance` for pre-built models
- Wrapping existing model instances
- Multiple forward passes with profiling

---

## 🖼️ Computer Vision

Real-world computer vision examples with popular architectures.

### ResNet & EfficientNet
```bash
kandc python examples/vision_models/resnet_example.py
```

**Architecture Highlights:**
- ResNet-18 with residual connections
- EfficientNet-style mobile architecture
- Batch normalization and skip connections

**Perfect for:**
- Image classification tasks
- Learning modern CNN architectures
- Comparing different model designs

**Expected Performance:**
- ResNet-18: ~11M parameters
- Multiple input sizes (224x224, 299x299)
- GPU utilization analysis

---

## 🤖 NLP & Transformers

Advanced natural language processing with transformer models.

### Custom Transformer Implementation
```bash
kandc python examples/nlp_models/transformer_example.py
```

**Models Included:**
- **GPT-like Model**: Autoregressive text generation
- **BERT-like Model**: Bidirectional encoder for MLM

**Technical Features:**
- Multi-head self-attention
- Positional encoding
- Causal and bidirectional attention masks
- Layer normalization and dropout

**What you'll learn:**
- Transformer architecture internals
- Attention mechanism profiling
- Memory usage patterns in large models

### HuggingFace Integration
```bash
kandc --requirements requirements_examples/nlp_requirements.txt -- python examples/nlp_models/pretrained_models.py
```

**Models Used:**
- DistilBERT for text classification
- Automatic model downloading and caching

**Production Features:**
- Real pretrained weights from HuggingFace Hub
- Tokenizer integration
- Text processing pipelines

### Large Model Downloads
```bash
kandc --requirements requirements_examples/nlp_requirements.txt -- python examples/nlp_models/huggingface_download_example.py
```

**What's Special:**
- Downloads actual model weights (~250MB DistilBERT, ~500MB GPT-2)
- Demonstrates large file handling
- Model caching for subsequent runs

---

## 🔍 Vision-Language Models

Cutting-edge multi-modal AI with vision-language understanding.

### CLIP Integration
```bash
kandc --requirements requirements_examples/vlm_requirements.txt -- python examples/vlm_models/clip_example.py
```

**Capabilities Demonstrated:**
- **Zero-shot Image Classification**: Classify images using text descriptions
- **Image-Text Similarity**: Compute semantic similarity scores
- **Real-world Image Processing**: Download and process web images

**Technical Highlights:**
- OpenAI CLIP model (~600MB download)
- Vision and text encoders
- Contrastive learning representations

**Use Cases:**
- Content moderation
- Image search and retrieval
- Multi-modal understanding

---

## 🎨 Generative Models

Advanced generative modeling with GANs and VAEs.

### GAN & VAE Implementation
```bash
kandc python examples/generative_models/gan_example.py
```

**Models Included:**
- **DCGAN**: Deep Convolutional GAN for image generation
- **VAE**: Variational Autoencoder with reparameterization trick

**Advanced Concepts:**
- Adversarial training dynamics
- Latent space manipulation
- Generator and discriminator profiling

**Perfect for:**
- Understanding generative modeling
- Profiling GAN training dynamics
- Comparing different generative approaches

---

## 🧪 Edge Cases & Testing

Specialized examples for testing Keys & Caches' capabilities.

### Command Line Arguments
```bash
# Interactive format
kandc python examples/edge_cases/model_with_args.py --model-size large --batch-size 16

# Separator format  
kandc --app-name "args-test" --gpu A100-80GB:2 -- python examples/edge_cases/model_with_args.py --model-size large --batch-size 16
```

**What you'll learn:**
- Both Keys & Caches command formats (interactive and separator)
- Script argument handling
- Configuration flexibility

### Long-Running Demo
```bash
# Quick demo (~1 minute)
kandc python examples/edge_cases/long_running_demo.py --epochs 3 --num-batches 8

# Full demo (~3 minutes)
kandc python examples/edge_cases/long_running_demo.py --epochs 8 --validate --verbose
```

**Features:**
- Real-time stdout streaming
- Progress tracking with visual indicators
- Configurable duration for testing
- Comprehensive logging

### Large File Handling
```bash
kandc python examples/edge_cases/large_file_test.py
```

**Testing:**
- Creates 100MB test file
- Demonstrates file caching system
- Upload optimization strategies

---

## 🏭 Production Workflows

Real-world scenarios and best practices.

### Multi-GPU Training Pattern
```python
# Scale from 1 to 8 GPUs seamlessly
@capture_model_class(model_name="ProductionModel")
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your large model architecture
        
# Commands for different scales:
# Development: 1 GPU
kandc --gpu A100-80GB:1 python train.py --debug

# Medium training: 2-4 GPUs  
kandc --gpu A100-80GB:4 python train.py --full-dataset

# Large scale: 8 GPUs
kandc --gpu H100:8 python train.py --full-dataset --large-batch
```

### Hyperparameter Sweeps
```bash
# Systematic exploration
kandc --app-name "sweep-lr-0001" --gpu A100-80GB:2 python train.py --lr 0.001
kandc --app-name "sweep-lr-0003" --gpu A100-80GB:2 python train.py --lr 0.003
kandc --app-name "sweep-lr-001" --gpu A100-80GB:2 python train.py --lr 0.01
```

### Model Comparison Pipeline
```bash
# Compare different architectures
kandc --app-name "resnet18-baseline" --gpu A100-80GB:1 python compare_models.py --arch resnet18
kandc --app-name "resnet50-comparison" --gpu A100-80GB:2 python compare_models.py --arch resnet50
kandc --app-name "efficientnet-test" --gpu H100:1 python compare_models.py --arch efficientnet
```

---

## 📊 Performance Analysis Examples

Learn to use Keys & Caches' profiling data effectively.

### Memory Usage Patterns
```python
@capture_model_class(model_name="MemoryAnalysis", profile_memory=True)
class AnalysisModel(nn.Module):
    # Model with intentional memory patterns
    def forward(self, x):
        # Large intermediate tensors
        x1 = self.layer1(x)  # Peak memory here
        x2 = self.layer2(x1)
        return x2
```

### Bottleneck Identification
```python
@capture_model_class(model_name="BottleneckFinder", record_shapes=True)
class SlowModel(nn.Module):
    def forward(self, x):
        # Intentionally slow operations for profiling
        x = self.expensive_conv(x)    # Bottleneck layer
        x = self.efficient_linear(x)  # Fast layer
        return x
```

### Batch Size Optimization
```bash
# Test different batch sizes
kandc --gpu L4:1 python batch_size_test.py --batch-size 16   # Memory efficient
kandc --gpu A100-80GB:1 python batch_size_test.py --batch-size 64   # Balanced
kandc --gpu H100:1 python batch_size_test.py --batch-size 256  # Maximum throughput
```

---

## 🎯 Specialized Use Cases

### Research & Experimentation
```bash
# Quick prototyping
kandc --gpu A100-80GB:1 python prototype.py --quick-test

# Full experiment
kandc --gpu H100:4 --app-name "paper-reproduction" python full_experiment.py
```

### Model Debugging
```bash
# Debug with detailed profiling
kandc --gpu A100-80GB:1 python debug_model.py --verbose --profile-memory
```

### Educational Examples
```bash
# For teaching ML concepts
kandc python examples/educational/attention_visualization.py
kandc python examples/educational/gradient_flow_analysis.py
```

---

## 📋 Requirements Templates

Use these as starting points for your projects:

### Minimal Setup
```txt
# requirements_examples/minimal_requirements.txt
torch>=2.0.0
numpy
```

### Computer Vision
```txt
# requirements_examples/vision_requirements.txt
torch>=2.0.0
torchvision
pillow
opencv-python
matplotlib
```

### NLP & Transformers
```txt
# requirements_examples/nlp_requirements.txt
torch>=2.0.0
transformers>=4.20.0
tokenizers
datasets
numpy
```

### Vision-Language Models
```txt
# requirements_examples/vlm_requirements.txt
torch>=2.0.0
transformers>=4.20.0
pillow
requests
numpy
matplotlib
```

### Scientific Computing
```txt
# requirements_examples/scientific_requirements.txt
torch>=2.0.0
numpy
scipy
matplotlib
seaborn
pandas
scikit-learn
```

---

## 🚀 Advanced Techniques

### Custom Model Architectures
```python
@capture_model_class(model_name="CustomArch")
class InnovativeModel(nn.Module):
    """Your cutting-edge architecture"""
    def __init__(self):
        super().__init__()
        # Novel architecture components
        
    def forward(self, x):
        # Custom forward pass logic
        return x
```

### Integration with MLOps Tools
```python
# Weights & Biases integration
import wandb

@capture_model_class(model_name="WandBIntegration")
class TrackedModel(nn.Module):
    def forward(self, x):
        output = self.layers(x)
        # Keys & Caches profiling + W&B logging
        wandb.log({"batch_processed": 1})
        return output
```

### Distributed Training Patterns
```bash
# Multi-node setup (coming soon)
kandc --gpu H100:8 --nodes 2 python distributed_train.py
```

---

## 💡 Best Practices Summary

### 🎯 **Development Workflow**
1. **Start Small**: Test with 1 GPU and small datasets
2. **Profile Early**: Use `@capture_model_class` from the beginning
3. **Scale Gradually**: 1 → 2 → 4 → 8 GPUs as needed
4. **Monitor Costs**: Check GPU utilization in traces

### 🔧 **Code Organization**
```
project/
├── models/           # Model definitions
├── data/            # Data loading utilities  
├── train.py         # Main training script
├── requirements.txt # Dependencies
└── .kandcignore    # Exclude unnecessary files
```

### 📊 **Performance Optimization**
- Use profiling data to identify bottlenecks
- Optimize batch sizes based on memory traces
- Compare different model architectures systematically
- Monitor GPU utilization across layers

### 🐛 **Debugging Strategy**
1. **Test Locally**: Always run `python script.py` first
2. **Check Logs**: Use real-time streaming to catch errors early
3. **Analyze Traces**: Use Chrome DevTools for visual debugging
4. **Start Simple**: Reduce complexity when debugging issues

---

## 🎉 Next Steps

Ready to build something amazing? Here are some ideas:

### 🚀 **Beginner Projects**
- Fine-tune a pretrained model on your dataset
- Build a custom CNN for image classification
- Experiment with different optimizers and learning rates

### 🔬 **Intermediate Projects**  
- Implement a paper from scratch
- Build a multi-modal model combining vision and text
- Create a custom loss function and compare performance

### 🏆 **Advanced Projects**
- Train a large language model from scratch
- Implement novel attention mechanisms
- Build a production-ready model serving pipeline

---

*Need help with any of these examples? Check out our [Contact & Support](contact.md) page for assistance!*
