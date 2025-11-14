#!/usr/bin/env python3
"""
TXT to PLY Converter
Converts TXT files containing point cloud data (X Y Z intensity) to PLY format.
Usage: python3 txt_ply_converter.py input_dir/ -o output_dir/
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path


def read_txt_file(txt_path):
    """
    Read TXT file and parse point cloud data.
    Expected format: X Y Z intensity (space-separated)
    
    Args:
        txt_path (str): Path to the TXT file
        
    Returns:
        numpy.ndarray: Array of points with shape (N, 4) where N is number of points
    """
    points = []
    
    try:
        with open(txt_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Split by whitespace and convert to float
                    values = line.split()
                    if len(values) >= 4:
                        x, y, z, intensity = map(float, values[:4])
                        points.append([x, y, z, intensity])
                    else:
                        print(f"Warning: Line {line_num} in {txt_path} has insufficient data: {line}")
                except ValueError as e:
                    print(f"Warning: Could not parse line {line_num} in {txt_path}: {line} - {e}")
                    
    except FileNotFoundError:
        print(f"Error: File not found: {txt_path}")
        return None
    except Exception as e:
        print(f"Error reading file {txt_path}: {e}")
        return None
    
    if not points:
        print(f"Warning: No valid points found in {txt_path}")
        return None
        
    return np.array(points)


def write_ply_file(points, output_path):
    """
    Write point cloud data to PLY format.
    
    Args:
        points (numpy.ndarray): Array of points with shape (N, 4)
        output_path (str): Path for the output PLY file
    """
    try:
        with open(output_path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("end_header\n")
            
            # Write point data
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f}\n")
                
    except Exception as e:
        print(f"Error writing PLY file {output_path}: {e}")
        return False
    
    return True


def convert_txt_to_ply(txt_path, output_path):
    """
    Convert a single TXT file to PLY format.
    
    Args:
        txt_path (str): Path to input TXT file
        output_path (str): Path for output PLY file
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    print(f"Converting: {txt_path} -> {output_path}")
    
    # Read TXT file
    points = read_txt_file(txt_path)
    if points is None:
        return False
    
    # Write PLY file
    success = write_ply_file(points, output_path)
    if success:
        print(f"Successfully converted {len(points)} points")
    
    return success


def batch_convert(input_dir, output_dir):
    """
    Convert all TXT files in input directory to PLY format in output directory.
    
    Args:
        input_dir (str): Input directory containing TXT files
        output_dir (str): Output directory for PLY files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all TXT files
    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"No TXT files found in {input_dir}")
        return False
    
    print(f"Found {len(txt_files)} TXT files to convert")
    
    success_count = 0
    for txt_file in txt_files:
        # Generate output filename (replace .txt with .ply)
        output_file = output_path / (txt_file.stem + ".ply")
        
        # Convert file
        if convert_txt_to_ply(str(txt_file), str(output_file)):
            success_count += 1
    
    print(f"\nConversion complete: {success_count}/{len(txt_files)} files converted successfully")
    return success_count == len(txt_files)


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert TXT files containing point cloud data to PLY format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 txt_ply_converter.py input_dir/ -o output_dir/
  python3 txt_ply_converter.py single_file.txt -o output_file.ply
        """
    )
    
    parser.add_argument("input", help="Input TXT file or directory containing TXT files")
    parser.add_argument("-o", "--output", required=True, help="Output PLY file or directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Single file conversion
        if not input_path.suffix.lower() == '.txt':
            print("Error: Input file must have .txt extension")
            sys.exit(1)
        
        # If output is a directory, create filename based on input
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / (input_path.stem + ".ply")
        
        success = convert_txt_to_ply(str(input_path), str(output_path))
        sys.exit(0 if success else 1)
        
    elif input_path.is_dir():
        # Directory batch conversion
        success = batch_convert(str(input_path), str(output_path))
        sys.exit(0 if success else 1)
        
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
