# Model Profiling with Chisel

Chisel provides powerful model profiling capabilities through the `@capture_model` decorator, which automatically profiles every forward pass of PyTorch models and provides detailed layer-level timing and shape analysis.

## Overview

The `@capture_model` decorator wraps PyTorch models to automatically:

- Profile every forward pass with PyTorch's profiler
- Capture layer-level timing information
- Track tensor shapes throughout the model
- Save detailed Chrome trace files
- Provide real-time layer summaries during execution

## Basic Usage

### Simple Model Profiling

```python
import torch
import torch.nn as nn
from chisel import capture_model

@capture_model(model_name="MyModel", record_shapes=True, profile_memory=True)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Create and use the model
model = MyModel()
data = torch.randn(32, 100)
output = model(data)  # This forward pass will be automatically profiled
```

### Complex Model Example

```python
@capture_model(model_name="ResNet", record_shapes=True, profile_memory=True)
class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

## Decorator Parameters

The `@capture_model` decorator accepts the following parameters:

### `model_name` (optional)
- **Type**: `str`
- **Default**: Model class name
- **Description**: Name used for trace files and analysis

### `record_shapes` (optional)
- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to record tensor shapes for each operation

### `profile_memory` (optional)
- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to profile memory usage

### Additional profiler arguments
- Any additional arguments are passed to PyTorch's profiler

## Trace Analysis

### Real-time Analysis

During execution on the Chisel backend, you'll see real-time layer summaries:

```
📊 Layer Analysis for MyModel
────────────────────────────────────────────────────────────
Layer                     Calls   Total Time (ms)   Avg Time (ms)   Shapes
────────────────────────────────────────────────────────────
linear1                   1       0.45              0.45            (32, 100)
relu                      1       0.12              0.12            (32, 50)
linear2                   1       0.23              0.23            (32, 50)
TOTAL                     3       0.80              0.27
```

### Post-execution Analysis

You can analyze trace files after execution using the `parse_model_trace` function:

```python
from chisel import parse_model_trace

# Analyze a trace file
analysis = parse_model_trace("MyModel_forward_001.json", "MyModel")

if analysis:
    layer_stats = analysis['layer_stats']
    
    # Print detailed analysis
    for layer_name, stats in layer_stats.items():
        total_time_ms = stats['total_time_us'] / 1000
        print(f"{layer_name}: {total_time_ms:.2f}ms")
        print(f"  Calls: {stats['call_count']}")
        print(f"  Shapes: {list(stats['shapes'])}")
        print(f"  Operations: {list(stats['operations'])}")
```

## Trace File Structure

Each forward pass generates a Chrome trace file with the naming pattern:
`{model_name}_forward_{pass_number:03d}.json`

### Trace File Contents

The trace files contain:

- **CPU Operations**: All PyTorch operations with timing
- **Shape Information**: Input/output shapes for each operation
- **Stack Traces**: Python call stack for operation identification
- **Memory Usage**: Memory allocation/deallocation events
- **GPU Operations**: CUDA kernel execution (if available)

### Example Trace Event

```json
{
  "ph": "X",
  "cat": "cpu_op",
  "name": "aten::linear",
  "dur": 450,
  "args": {
    "Input Dims": [[32, 100]],
    "Stack": [
      "model.py(15): forward",
      "linear.py(120): __call__"
    ]
  }
}
```

## Advanced Usage

### Multiple Forward Passes

The profiler automatically tracks multiple forward passes:

```python
model = MyModel()

# Each forward pass gets its own trace file
for i in range(5):
    output = model(data)  # Creates MyModel_forward_001.json, MyModel_forward_002.json, etc.
```

### Custom Model Names

```python
@capture_model(model_name="CustomResNet", record_shapes=True)
class MyResNet(nn.Module):
    # ... model definition
```

### Memory Profiling

```python
@capture_model(profile_memory=True)
class MemoryIntensiveModel(nn.Module):
    # ... model definition
```

## Integration with Chisel CLI

### Local Development

When running locally, the decorator acts as a pass-through:

```bash
python my_model_script.py  # No profiling, model works normally
```

### Cloud Execution

When running on Chisel backend, full profiling is activated:

```bash
chisel python my_model_script.py  # Full profiling with trace generation
```

## Best Practices

### 1. Model Design

- Use descriptive layer names for better analysis
- Structure models with clear forward() methods
- Avoid complex nested operations in forward()

### 2. Profiling Strategy

- Profile representative batch sizes
- Run multiple forward passes for statistical analysis
- Use different input shapes to understand scaling

### 3. Analysis

- Focus on layers with highest total time
- Look for shape mismatches or unexpected operations
- Compare timing across different model configurations

### 4. Performance Optimization

- Identify bottlenecks from layer timing
- Optimize layers with highest average time
- Consider model architecture changes based on profiling data

## Troubleshooting

### Common Issues

1. **No trace files generated**
   - Ensure you're running on Chisel backend (`CHISEL_BACKEND_RUN=1`)
   - Check that the model's forward() method is called

2. **Missing layer information**
   - Ensure `record_shapes=True` is set
   - Check that operations are PyTorch operations (not custom CUDA kernels)

3. **Incorrect layer names**
   - The profiler tries to extract layer names from stack traces
   - Complex nested operations may not be properly identified

### Debug Mode

Enable debug output by setting environment variables:

```bash
export CHISEL_DEBUG=1
chisel python my_model_script.py
```

## Examples

See the `examples/model_profiling_example.py` file for comprehensive examples including:

- Simple CNN models
- Transformer blocks
- Complex model architectures
- Trace analysis demonstrations

## API Reference

### `capture_model`

```python
def capture_model(
    model_name: Optional[str] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    **profiler_kwargs: Any,
) -> Callable
```

### `parse_model_trace`

```python
def parse_model_trace(
    trace_file: str,
    model_name: str = "Unknown"
) -> Optional[Dict]
```

Returns a dictionary with the following structure:

```python
{
    'model_name': str,
    'trace_file': str,
    'layer_stats': {
        'layer_name': {
            'total_time_us': int,
            'call_count': int,
            'shapes': set,
            'operations': set
        }
    }
}
```
