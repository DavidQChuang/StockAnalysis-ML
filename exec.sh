#!/bin/bash

# cd to this script's dir
cd "$(dirname "${BASH_SOURCE[0]}")"

# Include user vars
if [ -f "user-vars.sh" ]; then
    source user-vars.sh
else
    source user-vars.example.sh
fi

# Required Python version
REQUIRED_VERSION="3.11.0"

# Get the installed Python version
INSTALLED_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

# Compare versions
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$INSTALLED_VERSION" | sort -V | head -n1)" == "$REQUIRED_VERSION" ]]; then
    echo "Python version is greater than or equal to $REQUIRED_VERSION."
else
    echo "!! Python version is less than $REQUIRED_VERSION. Recommended version is >3.11."
fi

# Run script as module
python -m stockml.Trainer "$@"