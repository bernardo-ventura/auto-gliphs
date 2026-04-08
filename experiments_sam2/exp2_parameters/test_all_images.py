#!/usr/bin/env python3
"""
SAM 2 Parameter Testing - All Images
Tests different parameter configurations on ALL example images
Generates comparison visualizations like the SKIGO example for each image
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_anns(anns, ax):
    """Visualize all masks with random colors"""
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    ax.imshow(img)

def test_configuration(predictor, image_np, config_name, **kwargs):
    """Test a specific configuration"""
    # Create mask generator with custom parameters
    mask_generator = SAM2AutomaticMaskGenerator(predictor.model, **kwargs)
    
    # Generate masks
    masks = mask_generator.generate(image_np)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    return masks

def process_single_image(image_path, predictor, output_dir):
    """Process one image with all configurations"""
    image_name = Path(image_path).stem
    
    print(f"\n{'='*60}")
    print(f"📷 Processing: {image_name}")
    print(f"{'='*60}")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    print(f"   Size: {image_np.shape}")
    
    # Define configurations to test
    configs = [
        ("1_DEFAULT", {}),
        ("2_MORE_SENSITIVE", {
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9
        }),
        ("3_VERY_SENSITIVE", {
            'pred_iou_thresh': 0.7,
            'stability_score_thresh': 0.85
        }),
        ("4_MORE_POINTS", {
            'points_per_side': 48
        }),
        ("5_COMBINED_BEST", {
            'points_per_side': 48,
            'pred_iou_thresh': 0.75,
            'stability_score_thresh': 0.88
        })
    ]
    
    # Test all configurations
    results = []
    all_masks = []
    
    for config_name, params in configs:
        print(f"   Testing {config_name}... ", end='', flush=True)
        
        masks = test_configuration(predictor, image_np, config_name, **params)
        num_objects = len(masks)
        
        all_masks.append((config_name, masks))
        
        # Calculate stats
        if masks:
            avg_conf = np.mean([m['predicted_iou'] for m in masks])
            avg_stability = np.mean([m['stability_score'] for m in masks])
        else:
            avg_conf = 0
            avg_stability = 0
        
        result = {
            'config': config_name,
            'params': params,
            'num_objects': num_objects,
            'avg_confidence': float(avg_conf),
            'avg_stability': float(avg_stability)
        }
        results.append(result)
        
        print(f"✅ {num_objects} objects (conf: {avg_conf:.3f})")
    
    # Create comparison visualization
    create_comparison_grid(image_np, all_masks, image_name, output_dir)
    
    return {
        'image_name': image_name,
        'results': results
    }

def create_comparison_grid(image_np, all_masks, image_name, output_dir):
    """Create 2x3 grid visualization like SKIGO example"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create 2x3 grid
    # Row 1: Original, Config1, Config2
    # Row 2: Config3, Config4, Config5
    
    positions = [
        (0, 0),  # Original
        (0, 1),  # Config 1
        (0, 2),  # Config 2
        (1, 0),  # Config 3
        (1, 1),  # Config 4
        (1, 2),  # Config 5
    ]
    
    # Original image
    ax = plt.subplot2grid((2, 3), positions[0])
    ax.imshow(image_np)
    ax.set_title(f'{image_name}\nOriginal Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Each configuration
    for i, (config_name, masks) in enumerate(all_masks, 1):
        ax = plt.subplot2grid((2, 3), positions[i])
        ax.imshow(image_np)
        
        # Show masks
        show_anns(masks, ax)
        
        # Title with config name and object count
        num_objects = len(masks)
        ax.set_title(f'{config_name}\n{num_objects} objects', 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    viz_path = output_dir / f"{image_name}_parameter_comparison.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {viz_path.name}")

def main():
    print("="*60)
    print("SAM 2 PARAMETER TESTING - ALL IMAGES")
    print("="*60)
    
    # Configuration
    images_dir = Path("assets/examples")
    output_dir = Path("experiments_sam2/exp2_parameters/results")
    model_id = "facebook/sam2-hiera-tiny"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(images_dir.glob("*.png"))
    
    print(f"\n📁 Input directory: {images_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Found {len(image_files)} images")
    print(f"🤖 Model: {model_id}")
    
    # Load SAM 2
    print(f"\n📦 Loading SAM 2... ", end='', flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    print("✅")
    
    # Process all images
    all_results = []
    
    for image_path in image_files:
        result = process_single_image(image_path, predictor, output_dir)
        all_results.append(result)
    
    # Save consolidated results
    results_json = output_dir / "all_images_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}\n")
    
    print("Results by Configuration:\n")
    
    # Aggregate by configuration
    config_names = ["1_DEFAULT", "2_MORE_SENSITIVE", "3_VERY_SENSITIVE", 
                   "4_MORE_POINTS", "5_COMBINED_BEST"]
    
    for config in config_names:
        objects_list = []
        for img_result in all_results:
            for cfg_result in img_result['results']:
                if cfg_result['config'] == config:
                    objects_list.append(cfg_result['num_objects'])
        
        total = sum(objects_list)
        avg = total / len(objects_list) if objects_list else 0
        
        print(f"{config:20s}: Total: {total:3d} | Avg: {avg:4.1f} per image")
    
    # Best configuration per image
    print(f"\n{'='*60}")
    print("Best Configuration per Image:")
    print(f"{'='*60}\n")
    
    for img_result in all_results:
        best = max(img_result['results'], key=lambda x: x['num_objects'])
        print(f"{img_result['image_name']:15s}: {best['config']:20s} "
              f"({best['num_objects']} objects)")
    
    print(f"\n{'='*60}")
    print("✅ EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"\n💾 Outputs saved to: {output_dir}")
    print(f"   - Visualizations: *_parameter_comparison.png")
    print(f"   - Results data:   all_images_results.json")
    print(f"\n📊 Generated {len(image_files)} comparison grids!")

if __name__ == "__main__":
    main()
