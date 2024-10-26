#!/bin/bash

ENV_NAME="deepan"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    ENV_ACTIVATED_BY_SCRIPT=true
else
    ENV_ACTIVATED_BY_SCRIPT=false
fi

python "$@"

if [[ "$ENV_ACTIVATED_BY_SCRIPT" == true ]]; then
    conda deactivate
fi