# kandc Examples

This directory contains examples demonstrating all key features of the kandc library.

## Quick Start

### Prerequisites

```bash
pip install torch kandc
```

### Running the Examples

```bash
cd examples

# Full example with all features (requires authentication)
python complete_example.py

# Offline-only example (no internet required)
python offline_example.py
```

## Examples

### `complete_example.py` - Full Featured Demo

Showcases all kandc features:

**1. Different Modes**
- Online mode (default) - Full cloud experience with dashboard
- Offline mode - Local storage without internet
- Disabled mode - Zero overhead for production

**2. Experiment Tracking**
- `kandc.init()` - Initialize experiment with project and configuration
- `kandc.log()` - Log metrics, hyperparameters, and results
- `kandc.finish()` - Complete the experiment run

**3. Dashboard Features**
- Real-time metric visualization (online mode)
- Project organization and run comparison
- Local data storage (offline mode)


### `offline_example.py` - Offline Mode Demo

Perfect for development and environments without internet:

- **No authentication** - Works completely offline
- **Full profiling** - All model profiling features work locally
- **Local storage** - Data saved to `./kandc/project-name/run-id/`
- **CI/CD friendly** - Great for automated testing

## What You Get

### Online Mode
- 🔐 **Browser authentication** (first time only)
- 🌐 **Live dashboard** opens automatically
- ☁️ **Cloud sync** - All data backed up
- 📊 **Real-time charts** - Watch metrics update live
- 🔗 **Shareable links** - Collaborate with your team

### Offline Mode
- 📁 **Local files** - Everything saved to disk
- ✅ **Full profiling** - Models still analyzed completely
- ⚡ **Fast startup** - No network calls
- 🔒 **Private** - Data stays on your machine
- 🧪 **Development** - Perfect for debugging

### Disabled Mode
- 🚫 **Zero overhead** - All calls become no-ops
- 🚀 **Production ready** - No performance impact
- 📝 **Logging only** - Basic console output

## Example Output

When you run the examples, you'll see:

1. **Mode demonstration** - See all three modes in action
2. **Model profiling** - Automatic GPU/CPU analysis
3. **Timing collection** - Function performance measurement
4. **Metrics logging** - Real-time data tracking

In online mode, your dashboard opens automatically with:
- Detailed profiling traces viewable in Chrome
- Performance bottleneck analysis
- Metrics charts and comparisons
- Shareable experiment links

## Customization

You can modify the examples to:
- Use your own models and datasets
- Add custom metrics and logging
- Change profiling parameters
- Integrate with existing training code

## Next Steps

1. **Start with offline mode** if you want to try without authentication
2. **Run online mode** to see the full dashboard experience
3. **Adapt the patterns** to your own ML projects
4. **Check the documentation** for advanced features

The examples show you exactly how to integrate kandc into any PyTorch project!