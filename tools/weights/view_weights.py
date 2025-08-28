#!/usr/bin/env python3
"""
Tool to view specific weight tensors in safetensors files.

Usage:
    python view_weights.py <file_path> <tensor_name>
"""

import argparse
import sys
from pathlib import Path

try:
    from safetensors.torch import load_file
    import torch
except ImportError as e:
    print(f"Error: Missing required library. Install with: pip install safetensors torch")
    sys.exit(1)


def view_tensor(file_path: Path, tensor_name: str) -> None:
    """Load and display information about a specific tensor."""
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found")
        return
        
    try:
        tensors = load_file(str(file_path))
        
        if tensor_name not in tensors:
            print(f"Error: Tensor '{tensor_name}' not found in file")
            print(f"Available tensors: {list(tensors.keys())}")
            return
            
        tensor = tensors[tensor_name]
        print(f"🔍 Tensor: {tensor_name}")
        print(f"📐 Shape: {tuple(tensor.shape)}")
        print(f"🏷️  Type: {tensor.dtype}")
        print(f"🔢 Elements: {tensor.numel():,}")
        print(f"💾 Size: {tensor.numel() * tensor.element_size() / (1024*1024):.2f} MB")
        print(f"📊 Statistics:")
        print(f"   Min: {tensor.min().item():.6f}")
        print(f"   Max: {tensor.max().item():.6f}")
        print(f"   Mean: {tensor.mean().item():.6f}")
        print(f"   Std: {tensor.std().item():.6f}")
        
        # Show a few sample values
        if tensor.numel() > 0:
            flat_tensor = tensor.flatten()
            n_samples = min(10, tensor.numel())
            print(f"🎯 First {n_samples} values: {flat_tensor[:n_samples].tolist()}")
            
    except Exception as e:
        print(f"Error processing tensor: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View specific weight tensor in a safetensors file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.safetensors "model.embed_tokens.weight"
  %(prog)s pytorch_model.safetensors "lm_head.weight"
        """
    )
    parser.add_argument("file", type=Path, help="Path to the safetensors file")
    parser.add_argument("tensor", type=str, help="Name of the tensor to inspect")
    
    args = parser.parse_args()
    view_tensor(args.file, args.tensor)


if __name__ == "__main__":
    main()