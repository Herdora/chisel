#!/usr/bin/env python3
"""
Trace Analysis Example

This example demonstrates how to analyze the provided matrix_multiply.json trace file
and extract layer-level information from PyTorch profiler output.
"""

import json
import os
from pathlib import Path
from chisel import parse_model_trace


def analyze_provided_trace():
    """Analyze the provided matrix_multiply.json trace file."""
    print("🔍 Analyzing Provided Trace File")
    print("=" * 50)

    # Look for the matrix_multiply.json file
    trace_file = "matrix_multiply.json"

    if not os.path.exists(trace_file):
        print(f"❌ {trace_file} not found in current directory")
        print("💡 Please ensure the trace file is in the current directory")
        return

    print(f"📄 Found trace file: {trace_file}")

    # Use the parse_model_trace function
    analysis = parse_model_trace(trace_file, "MatrixMultiply")

    if analysis:
        print(f"✅ Successfully parsed trace file")
        print(f"📊 Model name: {analysis['model_name']}")

        layer_stats = analysis["layer_stats"]
        print(f"📊 Found {len(layer_stats)} layers/operations")

        # Print detailed analysis
        print(f"\n📋 Layer Analysis:")
        print("─" * 80)

        for layer_name, stats in layer_stats.items():
            total_time_ms = stats["total_time_us"] / 1000
            avg_time_ms = total_time_ms / stats["call_count"] if stats["call_count"] > 0 else 0

            print(f"\n🔹 {layer_name}")
            print(f"   📊 Total time: {total_time_ms:.2f}ms")
            print(f"   📊 Average time: {avg_time_ms:.2f}ms")
            print(f"   📊 Call count: {stats['call_count']}")

            if stats["shapes"]:
                print(f"   📐 Shapes: {list(stats['shapes'])}")

            if stats["operations"]:
                print(f"   ⚙️  Operations: {list(stats['operations'])}")

        # Calculate total model time
        total_model_time = sum(stats["total_time_us"] for stats in layer_stats.values()) / 1000
        print(f"\n📊 Total model execution time: {total_model_time:.2f}ms")

    else:
        print(f"❌ Failed to parse trace file")


def manual_trace_analysis():
    """Manually analyze the trace file to show the raw structure."""
    print(f"\n🔍 Manual Trace Analysis")
    print("=" * 50)

    trace_file = "matrix_multiply.json"

    if not os.path.exists(trace_file):
        print(f"❌ {trace_file} not found")
        return

    try:
        with open(trace_file, "r") as f:
            trace_data = json.load(f)

        events = trace_data.get("traceEvents", [])
        print(f"📊 Total events in trace: {len(events)}")

        # Count different event types
        event_types = {}
        cpu_ops = []

        for event in events:
            event_type = event.get("ph", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1

            # Collect CPU operations
            if event.get("ph") == "X" and event.get("cat") == "cpu_op":
                cpu_ops.append(event)

        print(f"\n📋 Event Type Breakdown:")
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count}")

        print(f"\n📊 CPU Operations: {len(cpu_ops)}")

        # Show some example CPU operations
        if cpu_ops:
            print(f"\n📋 Sample CPU Operations:")
            for i, op in enumerate(cpu_ops[:5]):  # Show first 5
                name = op.get("name", "unknown")
                dur = op.get("dur", 0)
                args = op.get("args", {})

                print(f"  {i + 1}. {name} ({dur}μs)")

                # Show input dimensions if available
                input_dims = args.get("Input Dims", [])
                if input_dims:
                    print(f"     Input shapes: {input_dims}")

                # Show stack info if available
                stack = args.get("Stack", [])
                if stack:
                    print(f"     Stack: {stack[:2]}...")  # Show first 2 stack frames

    except Exception as e:
        print(f"❌ Error reading trace file: {e}")


def demonstrate_shape_analysis():
    """Demonstrate shape analysis from the trace."""
    print(f"\n📐 Shape Analysis Demonstration")
    print("=" * 50)

    trace_file = "matrix_multiply.json"

    if not os.path.exists(trace_file):
        print(f"❌ {trace_file} not found")
        return

    try:
        with open(trace_file, "r") as f:
            trace_data = json.load(f)

        events = trace_data.get("traceEvents", [])

        # Collect all shapes from the trace
        all_shapes = set()
        shape_operations = {}

        for event in events:
            if event.get("ph") == "X" and event.get("cat") == "cpu_op":
                name = event.get("name", "")
                args = event.get("args", {})
                input_dims = args.get("Input Dims", [])

                if input_dims:
                    for dims in input_dims:
                        if dims and len(dims) > 0:
                            shape = tuple(dims)
                            all_shapes.add(shape)

                            if name not in shape_operations:
                                shape_operations[name] = set()
                            shape_operations[name].add(shape)

        print(f"📊 Total unique shapes found: {len(all_shapes)}")
        print(f"📊 Operations with shape info: {len(shape_operations)}")

        if all_shapes:
            print(f"\n📐 All Shapes:")
            for shape in sorted(all_shapes, key=lambda x: (len(x), x)):
                print(f"  {shape}")

        if shape_operations:
            print(f"\n📋 Shapes by Operation:")
            for op_name, shapes in shape_operations.items():
                print(f"  {op_name}: {list(shapes)}")

    except Exception as e:
        print(f"❌ Error analyzing shapes: {e}")


def main():
    """Main function to demonstrate trace analysis."""
    print("🚀 Trace Analysis Example")
    print("=" * 60)
    print(f"📍 Current directory: {os.getcwd()}")

    # Analyze the provided trace file
    analyze_provided_trace()

    # Show manual analysis
    manual_trace_analysis()

    # Demonstrate shape analysis
    demonstrate_shape_analysis()

    print(f"\n" + "=" * 60)
    print("✅ Trace Analysis Example Completed!")
    print("\n📚 What we learned:")
    print("  📊 How to parse PyTorch profiler trace files")
    print("  📐 How to extract shape information")
    print("  ⏱️  How to analyze timing data")
    print("  🔍 How to group operations by layer")
    print("\n💡 Next steps:")
    print("  📁 Use parse_model_trace() for automated analysis")
    print("  🔧 Customize analysis for your specific models")
    print("  📈 Build performance dashboards from trace data")
    print("=" * 60)


if __name__ == "__main__":
    main()
