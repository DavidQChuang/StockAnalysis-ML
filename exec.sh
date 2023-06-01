#!/bin/bash
source user-vars.sh

if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

"$python_cmd" Predictor.py "$@"