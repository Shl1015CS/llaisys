#!/usr/bin/env python3
"""
Tool to inspect tensor information in safetensors files.

Usage:
    python inspect_safetensors.py <file_path>
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors library not found. Install with: pip install safetensors")
    sys.exit(1)


def format_tensor_info(tensor) -> Dict[str, Any]:
    """Extract and format tensor information."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "size_mb": tensor.numel() * tensor.element_size() / (1024 * 1024)
    }


def inspect_safetensors(file_path: Path) -> None:
    """Inspect and display tensor information from safetensors file."""
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found")
        return
        
    try:
        with safe_open(file_path, framework="torch", device="cpu") as f:
            print(f"📂 File: {file_path}")
            print(f"🔢 Total tensors: {len(f.keys())}")
            print("=" * 80)
            
            total_params = 0
            total_size_mb = 0
            
            for name in sorted(f.keys()):
                tensor = f.get_tensor(name)
                info = format_tensor_info(tensor)
                
                total_params += info["numel"]
                total_size_mb += info["size_mb"]
                
                print(f"📊 {name:<40} {str(info['shape']):<20} {info['dtype']:<10} "
                      f"{info['numel']:>12,} params ({info['size_mb']:>6.2f} MB)")
            
            print("=" * 80)
            print(f"📈 Total parameters: {total_params:,}")
            print(f"💾 Total size: {total_size_mb:.2f} MB")
            
    except Exception as e:
        print(f"Error reading safetensors file: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect tensor information in safetensors files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.safetensors
  %(prog)s /path/to/model/pytorch_model.safetensors
        """
    )
    parser.add_argument("file", type=Path, help="Path to safetensors file")
    parser.add_argument("--sort", choices=["name", "size"], default="name", 
                       help="Sort tensors by name or size")
    
    args = parser.parse_args()
    inspect_safetensors(args.file)


if __name__ == "__main__":
    main()