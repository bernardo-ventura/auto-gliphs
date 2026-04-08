#!/usr/bin/env python3
"""
SAM 2 Parameter Testing
Tests different parameter configurations to improve segmentation quality
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
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Parameters: {kwargs}")
    
    # Create mask generator with custom parameters
    mask_generator = SAM2AutomaticMaskGenerator(predictor.model, **kwargs)
    
    # Generate masks
    masks = mask_generator.generate(image_np)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    print(f"✅ Detected {len(masks)} objects")
    
    # Show top 5
    if masks:
        print(f"Top 5 objects:")
        for i, mask in enumerate(masks[:5], 1):
            print(f"  {i}. Area: {mask['area']:5d} px | "
                  f"Conf: {mask['predicted_iou']:.3f} | "
                  f"Stability: {mask['stability_score']:.3f}")
    
    return masks, config_name

def main():
    print("="*60)
    print("SAM 2 PARAMETER TESTING")
    print("Testing different configurations to improve segmentation")
    print("="*60)
    
    # Load test image (SKIGO logo)
    image_path = "assets/examples/sample-0.png"
    output_dir = Path("experiments_sam2/exp2_parameters/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    print(f"\n📷 Test image: {image_path}")
    print(f"   Size: {image_np.shape}")
    
    # Load SAM 2
    print(f"\n📦 Loading SAM 2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    
    # Configuration presets - SIMPLIFIED VERSION (faster)
    configurations = [
        # 1. DEFAULT (baseline)
        {
            'name': '1_DEFAULT',
            'params': {}
        },
        
        # 2. MORE SENSITIVE - Lower thresholds
        {
            'name': '2_MORE_SENSITIVE',
            'params': {
                'pred_iou_thresh': 0.80,  # Default: 0.88 (lower = more permissive)
                'stability_score_thresh': 0.90  # Default: 0.95 (lower = more permissive)
            }
        },
        
        # 3. VERY SENSITIVE - Much lower thresholds
        {
            'name': '3_VERY_SENSITIVE',
            'params': {
                'pred_iou_thresh': 0.70,
                'stability_score_thresh': 0.85
            }
        },
        
        # 4. MORE POINTS - Denser sampling grid
        {
            'name': '4_MORE_POINTS',
            'params': {
                'points_per_side': 48,  # Default: 32 (more points = more objects detected)
            }
        },
        
        # 5. COMBINED - More points + lower thresholds (RECOMMENDED)
        {
            'name': '5_COMBINED_BEST',
            'params': {
                'points_per_side': 48,
                'pred_iou_thresh': 0.75,
                'stability_score_thresh': 0.88
            }
        }
    ]
    
    # Test all configurations
    all_results = []
    
    for config in configurations:
        masks, name = test_configuration(
            predictor, 
            image_np, 
            config['name'], 
            **config['params']
        )
        
        all_results.append({
            'name': name,
            'params': config['params'],
            'num_objects': len(masks),
            'masks': masks
        })
    
    # Create comparison visualization
    print(f"\n{'='*60}")
    print("Creating visualization...")
    
    n_configs = len(configurations)
    n_cols = 3
    n_rows = 2  # Fixed 2 rows for 6 plots
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()
    
    # First plot: original
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot each configuration
    for i, result in enumerate(all_results, 1):
        ax = axes[i]
        ax.imshow(image_np)
        show_anns(result['masks'], ax)
        
        # Title with number of objects
        title = f"{result['name']}\n{result['num_objects']} objects"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(all_results) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    viz_path = output_dir / "parameter_comparison.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved: {viz_path}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Configuration':<25} | {'Objects':<8} | Key Parameters")
    print("-" * 75)
    
    for result in sorted(all_results, key=lambda x: x['num_objects'], reverse=True):
        params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        if not params_str:
            params_str = "(default)"
        print(f"{result['name']:<25} | {result['num_objects']:<8} | {params_str}")
    
    # Save detailed results
    summary = {
        'image': image_path,
        'configurations': [
            {
                'name': r['name'],
                'parameters': r['params'],
                'num_objects': r['num_objects'],
                'top_objects': [
                    {
                        'area': int(m['area']),
                        'confidence': float(m['predicted_iou']),
                        'stability': float(m['stability_score'])
                    }
                    for m in r['masks'][:5]
                ]
            }
            for r in all_results
        ]
    }
    
    json_path = output_dir / "parameter_test_results.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Detailed results: {json_path}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("📚 PARAMETER GUIDE")
    print(f"{'='*60}\n")
    
    print("🔧 Key Parameters to Adjust:\n")
    
    print("1. points_per_side (default: 32)")
    print("   - Controls density of sampling grid")
    print("   - Higher = more objects detected")
    print("   - Try: 48, 64, 96 for finer detection\n")
    
    print("2. pred_iou_thresh (default: 0.88)")
    print("   - Minimum IoU to keep a mask")
    print("   - Lower = more permissive (accepts lower quality)")
    print("   - Try: 0.70-0.85 for more detections\n")
    
    print("3. stability_score_thresh (default: 0.95)")
    print("   - Minimum stability score")
    print("   - Lower = accepts less stable masks")
    print("   - Try: 0.80-0.90 for more objects\n")
    
    print("4. crop_n_layers (default: 0)")
    print("   - Number of crop layers for multi-scale")
    print("   - Higher = detects objects at different scales")
    print("   - Try: 1 or 2 for complex images\n")
    
    print("5. min_mask_region_area (default: 0)")
    print("   - Minimum area in pixels")
    print("   - Higher = filters out tiny objects")
    print("   - Try: 10-100 to remove noise\n")
    
    print("⚖️  Trade-offs:")
    print("   - More sensitive = more objects BUT more noise")
    print("   - Lower thresholds = better coverage BUT lower quality")
    print("   - More points = finer detail BUT slower processing")
    
    print(f"\n{'='*60}")
    print("✅ TESTING COMPLETED")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
