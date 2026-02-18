#!/bin/bash
# Quick demo script for Smart Home Motion Sensor System
# Run this to see all the features in action

echo "=================================================="
echo "Smart Home Motion Sensor System - Quick Demo"
echo "=================================================="
echo ""

# Change to the smart_home_sensors directory
cd "$(dirname "$0")"

# Ensure conda environment is activated
if ! command -v python &> /dev/null || ! python -c "import numpy" &> /dev/null 2>&1; then
    echo "Activating conda environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate habitat
fi

echo "1. Basic Motion Sensor Demo"
echo "   Running basic simulation with 3 rooms..."
echo ""
python smart_home_motion_sensors.py --demo
echo ""
echo "Press Enter to continue to next demo..."
read

echo ""
echo "=================================================="
echo "2. Multi-Person Tracking Demo"
echo "   Tracking 2 people moving through the house..."
echo ""
python smart_home_advanced.py --track-multiple-people
echo ""
echo "Press Enter to continue to next demo..."
read

echo ""
echo "=================================================="
echo "3. Activity Pattern Analysis Demo"
echo "   Analyzing morning routine patterns..."
echo ""
python smart_home_advanced.py --analyze-patterns
echo ""
echo "Press Enter to continue to test scenarios..."
read

echo ""
echo "=================================================="
echo "4. Test Scenarios"
echo ""
echo "Available scenarios:"
python test_motion_sensors.py --list-scenarios
echo ""
echo "Running 'corner detection' scenario..."
python test_motion_sensors.py --scenario corner
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "Running 'boundary detection' scenario..."
python test_motion_sensors.py --scenario boundary
echo ""

echo ""
echo "=================================================="
echo "✅ All demos complete!"
echo "=================================================="
echo ""
echo "To run individual demos:"
echo "  python smart_home_motion_sensors.py --demo"
echo "  python smart_home_advanced.py --track-multiple-people"
echo "  python test_motion_sensors.py --scenario <name>"
echo ""
echo "See SMART_HOME_SENSORS_README.md for full documentation"
echo ""
