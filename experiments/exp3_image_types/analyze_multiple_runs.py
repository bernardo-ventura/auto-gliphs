#!/usr/bin/env python3
"""
Comparative Analysis: Multiple Generations of Experiment 3

Creates a simple comparison file with:
- Generation time
- Complexity (number of SVG elements)
- Field for manual qualitative score (to be filled in)
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def load_results(results_dir):
    """Loads JSON results from a directory"""
    json_path = Path(results_dir) / "results.json"
    if not json_path.exists():
        return None
    
    with open(json_path) as f:
        return json.load(f)


def create_comparison_files(base_dir, all_runs, image_data):
    """Creates comparison files (CSV and Markdown)"""
    
    # 1. Create CSV for easy editing
    csv_path = base_dir / "comparison_table.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Image', 'Generation', 'Time (s)', 'Complexity (elements)', 
                  'Size (KB)', 'Valid', 'Qualitative_Score']
        writer.writerow(header)
        
        # Data
        for image_name in sorted(image_data.keys()):
            for data in sorted(image_data[image_name], key=lambda x: x['run']):
                row = [
                    image_name,
                    f"results{data['run']}",
                    round(data['generation_time'], 2),
                    data['num_elements'],
                    round(data['file_size_kb'], 2),
                    'Yes' if data['is_valid'] else 'No',
                    ''  # Empty field to fill manually
                ]
                writer.writerow(row)
    
    # 2. Create formatted Markdown
    md_path = base_dir / "comparison_table.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Comparison: Multiple Generations - Experiment 3\n\n")
        f.write("**Instructions:** Fill in the 'Qualitative Score' column with your subjective evaluation (e.g., 1-5, Poor/Good/Excellent, etc.)\n\n")
        
        for image_name in sorted(image_data.keys()):
            data_list = sorted(image_data[image_name], key=lambda x: x['run'])
            
            f.write(f"## 📷 {image_name}\n\n")
            f.write("| Generation | Time (s) | Complexity | Size (KB) | Valid | Qualitative Score |\n")
            f.write("|---------|-----------|--------------|--------------|--------|------------------|\n")
            
            for data in data_list:
                f.write(f"| results{data['run']} | "
                       f"{data['generation_time']:.2f} | "
                       f"{data['num_elements']} | "
                       f"{data['file_size_kb']:.2f} | "
                       f"{'✅' if data['is_valid'] else '❌'} | "
                       f"___ |\n")
            
            # Summary statistics row
            times = [d['generation_time'] for d in data_list]
            elements = [d['num_elements'] for d in data_list]
            
            f.write(f"| **Average** | "
                   f"**{sum(times)/len(times):.2f}** | "
                   f"**{sum(elements)/len(elements):.1f}** | "
                   f"— | — | — |\n")
            f.write(f"| **Range** | "
                   f"{min(times):.2f} - {max(times):.2f} | "
                   f"{min(elements)} - {max(elements)} | "
                   f"— | — | — |\n")
            f.write("\n")
    
    return csv_path, md_path


def analyze_runs():
    """Analyzes multiple runs and compares results"""
    
    base_dir = Path("experiments/exp3_image_types")
    
    # Load all results
    all_runs = {}
    for i in range(1, 6):
        results_dir = base_dir / f"results{i}"
        if results_dir.exists():
            results = load_results(results_dir)
            if results:
                all_runs[i] = results
    
    if not all_runs:
        print("❌ No results found!")
        return
    
    print("="*70)
    print("📊 CREATING COMPARISON FILE")
    print("="*70)
    print(f"\n🔢 Runs found: {len(all_runs)}")
    print(f"   Folders: {', '.join(f'results{i}' for i in sorted(all_runs.keys()))}")
    
    # Organize data by image
    image_data = defaultdict(list)
    
    for run_id, results in all_runs.items():
        for result in results:
            if result['success']:
                image_name = result['image_name']
                image_data[image_name].append({
                    'run': run_id,
                    'generation_time': result['generation_time'],
                    'num_elements': result['stats']['num_elements'],
                    'num_paths': result['stats']['num_paths'],
                    'file_size_kb': result['stats']['file_size_kb'],
                    'is_valid': result['stats']['is_valid']
                })
    
    # Create comparison files
    csv_path, md_path = create_comparison_files(base_dir, all_runs, image_data)
    
    # Visual summary
    print(f"\n{'='*70}")
    print("📸 DATA SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Total images: {len(image_data)}")
    print(f"Total generations per image: {len(all_runs)}")
    print(f"Total comparisons: {sum(len(data) for data in image_data.values())}")
    
    # Show data preview
    print(f"\n📋 PREVIEW (first 3 images):")
    for i, image_name in enumerate(sorted(image_data.keys())[:3], 1):
        data = sorted(image_data[image_name], key=lambda x: x['run'])
        print(f"\n{i}. {image_name}")
        for d in data:
            print(f"   results{d['run']}: {d['generation_time']:.2f}s | {d['num_elements']} elements")
    
    print(f"\n{'='*70}")
    print("✅ FILES CREATED")
    print(f"{'='*70}\n")
    print(f"📄 CSV:      {csv_path}")
    print(f"📄 Markdown: {md_path}")
    print(f"\n💡 Next steps:")
    print(f"   1. Open the SVG files in each results1-5 folder")
    print(f"   2. Compare visually with the original image")
    print(f"   3. Fill in the 'Qualitative_Score' column in the CSV or Markdown")
    print(f"   4. Use the data to draw conclusions about model consistency")


if __name__ == "__main__":
    analyze_runs()
