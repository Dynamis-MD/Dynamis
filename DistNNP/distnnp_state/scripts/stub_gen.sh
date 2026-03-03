#!/bin/bash

# 1. Navigate to the project root (one level up from the scripts directory)
# This ensures paths like "./src/" always resolve correctly, 
# even if you run the script from a different folder.
cd "$(dirname "$0")/.." || exit

echo "Generating stubs for distnnp_state_impl..."

# 2. Run the stub generator targeting the project root's src folder
pybind11-stubgen distnnp_state.distnnp_state_impl \
    -o ./src/ \
    --ignore-invalid-expressions ".*"

# 3. Ensure the py.typed marker exists
touch ./src/distnnp_state/py.typed

echo "Success! Stubs generated in src/distnnp_state/"