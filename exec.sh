#!/bin/bash

if [ -f "user-vars.sh" ]; then
    source user-vars.sh
else
    source user-vars.example.sh
fi

if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

"$python_cmd" Trainer.py "$@"