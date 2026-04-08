#!/usr/bin/env python3
"""
SAM 2 Consistency Analysis - Single Run
Runs SAM 2 segmentation once and saves results for later comparison
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
import argparse
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

def process_image(image_path, mask_generator, output_dir, run_number):
    """Process a single image with SAM 2"""
    image_name = Path(image_path).stem
    
    print(f"  📷 {image_name}... ", end='', flush=True)
    
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Generate masks
    try:
        masks = mask_generator.generate(image_np)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        num_objects = len(masks)
        print(f"✅ {num_objects} objects", flush=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image_np)
    axes[0].set_title(f'{image_name}\nOriginal')
    axes[0].axis('off')
    
    axes[1].imshow(image_np)
    show_anns(masks, axes[1])
    axes[1].set_title(f'SAM 2 - Run {run_number}\n{len(masks)} objects')
    axes[1].axis('off')
    
    plt.tight_layout()
    viz_path = output_dir / f"{image_name}_run{run_number}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Prepare results
    results = {
        'image_name': image_name,
        'run': run_number,
        'num_objects': len(masks),
        'objects': []
    }
    
    for i, mask_data in enumerate(masks, 1):
        obj_info = {
            'id': i,
            'area': int(mask_data['area']),
            'bbox': [int(x) for x in mask_data['bbox']],
            'confidence': float(mask_data['predicted_iou']),
            'stability': float(mask_data['stability_score']),
            'bbox_center': [
                int(mask_data['bbox'][0] + mask_data['bbox'][2] / 2),
                int(mask_data['bbox'][1] + mask_data['bbox'][3] / 2)
            ]
        }
        results['objects'].append(obj_info)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='SAM 2 consistency test - single run')
    parser.add_argument('--run', type=int, required=True, help='Run number (1-5)')
    args = parser.parse_args()
    
    run_number = args.run
    
    print("="*60)
    print(f"SAM 2 CONSISTENCY TEST - RUN {run_number}/5")
    print("="*60)
    
    # Configuration
    images_dir = Path("assets/examples")
    output_base = Path("experiments_sam2/exp_multiple_images")
    output_dir = output_base / f"results{run_number}"
    model_id = "facebook/sam2-hiera-tiny"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(images_dir.glob("*.png"))
    
    print(f"\n📊 Processing {len(image_files)} images")
    print(f"💾 Output: {output_dir}")
    
    # Load SAM 2
    print(f"\n📦 Loading SAM 2... ", end='', flush=True)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = SAM2ImagePredictor.from_pretrained(model_id)
        mask_generator = SAM2AutomaticMaskGenerator(predictor.model)
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Process all images
    print(f"\n🔍 Segmenting images:")
    all_results = []
    
    for image_path in image_files:
        result = process_image(image_path, mask_generator, output_dir, run_number)
        if result:
            all_results.append(result)
    
    # Save results
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Quick summary
    total_objects = sum(r['num_objects'] for r in all_results)
    avg_objects = total_objects / len(all_results) if all_results else 0
    
    print(f"\n{'='*60}")
    print(f"✅ Run {run_number} completed")
    print(f"📊 Total objects: {total_objects} | Average: {avg_objects:.1f} per image")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
