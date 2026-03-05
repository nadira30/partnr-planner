#!/bin/bash
# Test script for batch object addition API

EPISODE_ID="${1:-100}"
YAML_FILE="${2:-example_objects.yaml}"
API_URL="http://localhost:5002"

echo "=================================================="
echo "Batch Add Objects Test"
echo "=================================================="
echo "Episode ID: $EPISODE_ID"
echo "YAML File: $YAML_FILE"
echo ""

# Check if YAML file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: YAML file '$YAML_FILE' not found!"
    echo ""
    echo "Usage: $0 [episode_id] [yaml_file]"
    echo "Example: $0 100 my_objects.yaml"
    exit 1
fi

echo "YAML Content:"
echo "--------------------------------------------------"
cat "$YAML_FILE"
echo "--------------------------------------------------"
echo ""

echo "Sending request to API..."
echo ""

# Make the API request
response=$(curl -s -X POST \
    "$API_URL/api/episode/$EPISODE_ID/add-objects-batch" \
    -F "file=@$YAML_FILE")

# Pretty print the response
echo "API Response:"
echo "--------------------------------------------------"
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
echo "--------------------------------------------------"
echo ""

# Parse success status
success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null)

if [ "$success" = "True" ]; then
    echo "✓ All objects added successfully!"
else
    echo "⚠ Some objects failed to add. Check results above."
fi

echo ""
echo "To view the updated episode, visit:"
echo "  $API_URL/?episode_id=$EPISODE_ID"
echo ""
