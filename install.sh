#!/bin/bash

# cd to this script's dir
cd "$(dirname "${BASH_SOURCE[0]}")"

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
    echo "!! Python version is less than $REQUIRED_VERSION. This may cause issues."
fi

# Set the python command if not given by environment var 'python_cmd'
if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

# Create venv if directory doesn't exist.
if [[ ! -d "venv" ]]
then
    "$python_cmd" -m venv venv || { echo "venv should be installed. Delete the venv directory and run 'pip3 install venv' before running this script again." ; exit 1; }
    source venv/bin/activate
    "$python_cmd" -m pip install -r requirements.txt
fi

# Run script from root directory
# "$python_cmd" stockml/Trainer.py "$@"