#!/usr/bin/env python3
"""
SAM 2 Experiment: Multiple Images Segmentation
Tests SAM 2 automatic segmentation on all example images
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

def process_image(image_path, mask_generator, output_dir):
    """Process a single image with SAM 2"""
    image_name = Path(image_path).stem
    
    print(f"\n{'='*60}")
    print(f"📷 Processing: {image_name}")
    
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        print(f"   Size: {image_np.shape}")
    except Exception as e:
        print(f"   ❌ Error loading image: {e}")
        return None
    
    # Generate masks
    print(f"   🔍 Detecting objects...")
    try:
        masks = mask_generator.generate(image_np)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        print(f"   ✅ Found {len(masks)} objects")
    except Exception as e:
        print(f"   ❌ Error during segmentation: {e}")
        return None
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image_np)
    axes[0].set_title(f'{image_name}\nOriginal')
    axes[0].axis('off')
    
    axes[1].imshow(image_np)
    show_anns(masks, axes[1])
    axes[1].set_title(f'SAM 2 Segmentation\n{len(masks)} objects')
    axes[1].axis('off')
    
    plt.tight_layout()
    viz_path = output_dir / f"{image_name}_segmentation.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {viz_path.name}")
    
    # Prepare results (without the actual mask arrays for JSON)
    results = {
        'image_name': image_name,
        'image_path': str(image_path),
        'image_size': list(image_np.shape),
        'num_objects': len(masks),
        'objects': []
    }
    
    for i, mask_data in enumerate(masks, 1):
        obj_info = {
            'id': i,
            'area': int(mask_data['area']),
            'bbox': [int(x) for x in mask_data['bbox']],  # [x, y, w, h]
            'confidence': float(mask_data['predicted_iou']),
            'stability': float(mask_data['stability_score'])
        }
        results['objects'].append(obj_info)
    
    # Show top 5 objects
    if len(masks) > 0:
        print(f"   📊 Top 5 objects:")
        for i, obj in enumerate(results['objects'][:5], 1):
            print(f"      {i}. Area: {obj['area']:4d} px | "
                  f"BBox: ({obj['bbox'][0]:3d}, {obj['bbox'][1]:3d}, "
                  f"{obj['bbox'][2]:3d}, {obj['bbox'][3]:3d}) | "
                  f"Conf: {obj['confidence']:.3f}")
    
    return results

def main():
    print("="*60)
    print("SAM 2 EXPERIMENT: Multiple Images Segmentation")
    print("="*60)
    
    # Configuration
    images_dir = Path("assets/examples")
    output_dir = Path("experiments_sam2/exp_multiple_images/results")
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
    print(f"\n📦 Loading SAM 2...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
        
        # Load predictor and create mask generator
        predictor = SAM2ImagePredictor.from_pretrained(model_id)
        mask_generator = SAM2AutomaticMaskGenerator(predictor.model)
        
        print("   ✅ SAM 2 loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading SAM 2: {e}")
        return
    
    # Process all images
    all_results = []
    
    for image_path in image_files:
        result = process_image(image_path, mask_generator, output_dir)
        if result:
            all_results.append(result)
    
    # Save consolidated results
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    total_images = len(all_results)
    total_objects = sum(r['num_objects'] for r in all_results)
    avg_objects = total_objects / total_images if total_images > 0 else 0
    
    print(f"\n✅ Processed: {total_images} images")
    print(f"📊 Total objects detected: {total_objects}")
    print(f"📈 Average objects per image: {avg_objects:.1f}")
    
    # Show results per image
    print(f"\n📋 Results per image:")
    for result in sorted(all_results, key=lambda x: x['num_objects'], reverse=True):
        print(f"   {result['image_name']:12s}: {result['num_objects']:2d} objects")
    
    # Object size statistics
    all_areas = [obj['area'] for r in all_results for obj in r['objects']]
    if all_areas:
        print(f"\n📏 Object size statistics:")
        print(f"   Smallest: {min(all_areas)} pixels")
        print(f"   Largest:  {max(all_areas)} pixels")
        print(f"   Average:  {sum(all_areas)/len(all_areas):.1f} pixels")
    
    # Confidence statistics
    all_confidences = [obj['confidence'] for r in all_results for obj in r['objects']]
    if all_confidences:
        print(f"\n🎯 Confidence statistics:")
        print(f"   Min:     {min(all_confidences):.3f}")
        print(f"   Max:     {max(all_confidences):.3f}")
        print(f"   Average: {sum(all_confidences)/len(all_confidences):.3f}")
    
    print(f"\n{'='*60}")
    print("✅ EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"\n💾 Outputs saved to: {output_dir}")
    print(f"   - Visualizations: *_segmentation.png")
    print(f"   - Results data:   results.json")
    print(f"\n💡 Next step: Integrate with StarVector for SVG annotation!")

if __name__ == "__main__":
    main()
