#!/usr/bin/env python3
"""
Example demonstrating the profiler system in kandc.

This example shows how to use both the ProfilerWrapper and ProfilerDecorator
to profile method calls and function execution times, integrated with the
full kandc experiment tracking system.
"""

import time
import random
from kandc import init, finish
from kandc.annotators import ProfilerWrapper, ProfilerDecorator, profile, profiler


# Example 1: Using ProfilerWrapper to wrap an existing instance
class MyModel:
    """A simple model class for demonstration."""
    
    def __init__(self, name="MyModel"):
        self.name = name
        self.cache = {}
    
    def preprocess(self, data):
        """Simulate preprocessing step."""
        time.sleep(0.01)  # Simulate work
        return f"processed_{data}"
    
    def forward(self, input_data):
        """Simulate forward pass."""
        time.sleep(0.02)  # Simulate work
        if input_data in self.cache:
            return self.cache[input_data]
        
        # Simulate some computation
        result = self.preprocess(input_data)
        result = self._internal_compute(result)
        self.cache[input_data] = result
        return result
    
    def _internal_compute(self, data):
        """Internal method that gets called by forward."""
        time.sleep(0.005)  # Simulate work
        return f"computed_{data}"
    
    def batch_inference(self, batch):
        """Simulate batch processing."""
        time.sleep(0.05)  # Simulate work
        results = []
        for item in batch:
            results.append(self.forward(item))
        return results


# Example 2: Using ProfilerDecorator on a class
@ProfilerDecorator("OptimizedModel")
class OptimizedModel:
    """A model class that gets profiling automatically."""
    
    def __init__(self):
        self.cache = {}
    
    def predict(self, x):
        """Make a prediction."""
        time.sleep(0.015)  # Simulate work
        return f"prediction_{x}"
    
    def train(self, data):
        """Train the model."""
        time.sleep(0.1)  # Simulate work
        return f"trained_on_{len(data)}_samples"


# Example 3: Using ProfilerDecorator on a function
@profiler("expensive_function")
def expensive_function(n):
    """A function that gets profiled automatically."""
    time.sleep(0.03)  # Simulate work
    return sum(i * i for i in range(n))


def demonstrate_wrapper():
    """Demonstrate using ProfilerWrapper."""
    print("🔧 Example 1: Using ProfilerWrapper")
    print("=" * 50)
    
    # Create a regular instance
    model = MyModel("TestModel")
    
    # Wrap it with profiling
    profiled_model = ProfilerWrapper(model, "TestModel")
    
    # Make some calls
    print("Making method calls...")
    result1 = profiled_model.forward("input1")
    result2 = profiled_model.forward("input2")
    result3 = profiled_model.batch_inference(["input3", "input4", "input5"])
    
    print(f"Results: {result1}, {result2}, {result3}")
    
    # Print profiling summary
    profiled_model.print_summary()
    
    return profiled_model


def demonstrate_decorator():
    """Demonstrate using ProfilerDecorator on a class."""
    print("\n🔧 Example 2: Using ProfilerDecorator on a class")
    print("=" * 50)
    
    # Create instances - they get profiling automatically
    model1 = OptimizedModel()
    model2 = OptimizedModel()
    
    print("Making method calls on decorated class...")
    result1 = model1.predict("sample1")
    result2 = model2.predict("sample2")
    result3 = model1.train(["data1", "data2", "data3"])
    
    print(f"Results: {result1}, {result2}, {result3}")
    
    # Note: The decorator approach doesn't provide easy access to stats
    # since it modifies the class itself. For detailed stats, use the wrapper approach.


def demonstrate_function_decorator():
    """Demonstrate using ProfilerDecorator on a function."""
    print("\n🔧 Example 3: Using ProfilerDecorator on a function")
    print("=" * 50)
    
    print("Calling decorated function...")
    result1 = expensive_function(1000)
    result2 = expensive_function(2000)
    
    print(f"Results: {result1}, {result2}")


def demonstrate_convenience_functions():
    """Demonstrate using convenience functions."""
    print("\n🔧 Example 4: Using convenience functions")
    print("=" * 50)
    
    # Using the profile() function
    model = MyModel("ConvenienceModel")
    profiled = profile(model, "ConvenienceModel")
    
    print("Making calls with convenience wrapper...")
    result = profiled.forward("test_input")
    print(f"Result: {result}")
    
    # Print stats
    profiled.print_summary()


def demonstrate_environment_control():
    """Demonstrate how environment variables control profiling."""
    print("\n🔧 Example 5: Environment Variable Control")
    print("=" * 50)
    
    import os
    
    print(f"Current KANDC_PROFILER_DISABLED: {os.environ.get('KANDC_PROFILER_DISABLED', 'Not set')}")
    
    if os.environ.get('KANDC_PROFILER_DISABLED'):
        print("Profiling is disabled by environment variable")
    else:
        print("Profiling is enabled (default behavior)")
    
    print("\nTo disable profiling, set: export KANDC_PROFILER_DISABLED=1")
    print("To enable profiling, unset: unset KANDC_PROFILER_DISABLED")





def main():
    """Run all demonstrations."""
    print("🚀 kandc Profiler System Demonstration")
    print("=" * 60)
    
    try:
        # Initialize kandc experiment tracking
        print("🔧 Initializing kandc experiment tracking...")
        init("profiler_demo", name="profiler_demo", description="Demonstrating the profiler system")
        
        demonstrate_wrapper()
        demonstrate_decorator()
        demonstrate_function_decorator()
        demonstrate_convenience_functions()
        demonstrate_environment_control()
        
        print("\n✅ All demonstrations completed!")
        print("\n💡 Key Features:")
        print("   • Automatic method interception with reentrancy protection")
        print("   • Both wrapper and decorator approaches")
        print("   • Environment variable control (KANDC_PROFILER_DISABLED)")
        print("   • Backend integration for persistent logging")
        print("   • Real-time console output and statistics")
        print("   • Full integration with kandc experiment tracking")
        print("   • Automatic PyTorch trace creation and artifact upload")
        
        # Finish the experiment
        print("\n🔧 Finishing kandc experiment...")
        finish()
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to finish even if there's an error
        try:
            finish()
        except:
            pass


if __name__ == "__main__":
    main()
