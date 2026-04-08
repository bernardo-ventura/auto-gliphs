#!/bin/bash
# Run SAM 2 segmentation 5 times for consistency analysis

echo "Running SAM 2 segmentation 5 times for each image..."

for i in {1..5}; do
    echo ""
    echo "================================"
    echo "RUN $i of 5"
    echo "================================"
    python experiments_sam2/exp_multiple_images/analyze_consistency.py --run $i
done

echo ""
echo "✅ All 5 runs completed!"
echo "Run analysis script to compare results:"
echo "python experiments_sam2/exp_multiple_images/analyze_results.py"
