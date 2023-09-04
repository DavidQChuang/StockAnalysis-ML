#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -f "user-vars.sh" ]; then
    source user-vars.sh
else
    source user-vars.example.sh
fi

if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

if [[ ! -d "venv" ]]
then
    "$python_cmd" -m venv venv || echo "venv should be installed. Run 'pip install venv'."
    source venv/bin/activate
    pip install -r requirements.txt
fi

"$python_cmd" Trainer.py "$@"