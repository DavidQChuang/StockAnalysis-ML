# ML Stock Analysis
## Usage
Set environment variables in user-vars.sh, then call exec.sh. Arguments are passed through to the python program (see below for arguments).

### Runs
'Runs' describe an NN architecture and dataset, which are used to perform time-series inference on the described stock or digital currency data. Runs will be read from `runs/model_runs.json` by default (see -rf below).

Runs are read from the JSON file in the following order after a run is selected:
1. Environment variables are read from `'env'`, which must be an object, and are then merged into the rest of the file in the position described by the key. For example, the entry `"global.other.etc.apikey": "OTHER_APIKEY"` below will be merged into the nested dictionary `global -> other -> etc["apikey"]`, and set to the value of the environment variable `OTHER_APIKEY`. Environment variables will not change values that already exist.
2. The selected run (one of the JSON objects within `'runs'`, such as `'run-1'` below) is read, then the global run (the JSON object under `'global'`) is merged into it. Global values will be overridden by local run values.
3. Within the run, model.seq_len and out_seq_len are copied into dataset.seq_len and dataset.out_seq_len. 

Example:
```
# In runs/model_runs.json:
{
    ...
    "env": {
        "global.data_retriever.alphavantage.apikey": "ALPHAVANTAGE_APIKEY"
        "global.other.etc.apikey": "OTHER_APIKEY"
    },
    "global": { ... }
    "runs": {
        "run-1": { ... }
    }
}
# See runs/sample_runs.json for sample runs.

# In user-vars.sh:
export ALPHAVANTAGE_APIKEY=XXXX
export OTHER_APIKEY=XXXX
# export python_cmd="python3"

# Then call exec.sh with command line args to run the program.
$ exec.sh -r run-1
```
## Arguments
`-r, --run-name`: Name of the run to use. If this doesn't match a run name, it will attempt to match the start of a run name, and if only one is found, it will use that run.

`-rf, --run-file`: Path of the file to load the runs from. Default is `runs/model_runs.json`.

`-mf, --model-file`: Path of the file to save/load the model from. 
By default, the model will determine a filename automatically based on architecture parameters, such as `SimpleLSTM-256_72+1.ckpt`.

`-rm, --rebuild-model`: If true then overwrites old model and starts from scratch.
Default behavior is to continue training with the existing checkpoint.

`-d, --device`: Specifies the device to use. Possible values: `cpu`, `cuda`.

`-v, --verbosity`: (not implemented)

0: quiet - only run selection, final metrics and trailing predicted prices will be printed.

1: default - the above + announcing each step, and stating basic operations and statistics such as the validation split and number of data rows, and small data previews.

2: diagnostic - the above + model summary, 