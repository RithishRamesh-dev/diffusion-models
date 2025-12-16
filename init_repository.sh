#!/bin/bash

echo "Initializing diffusion-models repository..."

# Create directories
mkdir -p wan_22/outputs
mkdir -p flux_2/outputs

# Create .gitignore
cat > .gitignore << 'EOF'
# Outputs
wan_22/outputs/*.mp4
flux_2/outputs/*

# Model cache
hf_models/

# Python
__pycache__/
*.pyc

# System
.DS_Store
EOF

# Make scripts executable
chmod +x wan_22/run.sh 2>/dev/null || true

echo "Done! Repository structure created."
echo ""
echo "Next steps:"
echo "  cd wan_22"
echo "  ./run.sh"