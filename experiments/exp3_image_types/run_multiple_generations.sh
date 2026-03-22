#!/bin/bash
# Script to run multiple generations of experiment 3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_SCRIPT="$SCRIPT_DIR/exp3_image_types.py"

# Images used in results1
IMAGES=(
    "assets/examples/sample-0.png"
    "assets/examples/sample-1.png"
    "assets/examples/sample-4.png"
    "assets/examples/sample-6.png"
    "assets/examples/sample-7.png"
    "assets/examples/sample-15.png"
    "assets/examples/sample-16.png"
    "assets/examples/sample-17.png"
    "assets/examples/sample-18.png"
)

echo "=========================================="
echo "EXPERIMENT 3: MULTIPLE GENERATIONS"
echo "=========================================="
echo ""
echo "Running 4 additional generations..."
echo "Output folders: results2, results3, results4, results5"
echo ""

# Run generations 2, 3, 4 and 5
for run in {2..5}; do
    echo ""
    echo "=========================================="
    echo "🔄 GENERATION $run/5"
    echo "=========================================="
    
    OUTPUT_DIR="$SCRIPT_DIR/results$run"
    
    python "$EXP_SCRIPT" \
        --images "${IMAGES[@]}" \
        --output-dir "$OUTPUT_DIR"
    
    echo ""
    echo "✅ Generation $run completed: $OUTPUT_DIR"
done

echo ""
echo "=========================================="
echo "✅ ALL GENERATIONS COMPLETED"
echo "=========================================="
echo ""
echo "📂 Results saved in:"
echo "   - experiments/exp3_image_types/results1 (original)"
echo "   - experiments/exp3_image_types/results2"
echo "   - experiments/exp3_image_types/results3"
echo "   - experiments/exp3_image_types/results4"
echo "   - experiments/exp3_image_types/results5"
echo ""
echo "💡 Next step: Run the comparative analysis script"
echo "   python experiments/exp3_image_types/analyze_multiple_runs.py"
