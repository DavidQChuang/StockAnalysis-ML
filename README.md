# ML Stock Analysis
## Capabilities
| Feature               | Usage             | Implemented?  |
| -------------         | -------------     | ---           |
| **Predictive** | For predicting future market performance. |
| Simple Inference      | Predicts the value of the next datapoint based on market data. | Yes |
| Sentiment Analysis         | Analyzes the sentiment of the market (bear or bull) based on market data.  | No |
| News Sentiment Analysis    | Predicts the effect of news events on sentiment (negative or positive) and the magnitude of the resulting change in market price for each time interval, based on market data and language analysis. | No |
| **Analytical** | For deriving additional meaning from market data, improving performance of predictive features.|
| Sentiment Pattern Matching | Compares windows of market data, classifying each window in terms of market sentiment (bear or bull). | No |
| PCA | Reduces the dimensionality of the data using eigenvectors. | No |
| **Automatic** | For deriving action from market data and predictive/analytical data. |
| Automatic Trader | Sends specific buy and sell orders based on market data, predictive/analytical data, and account data. | No |

## Usage
Set environment variables in .env, then call `uv run stockml-train <args>`. Arguments are passed through to the python program (see below for arguments).

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

# In .env:
export ALPHAVANTAGE_APIKEY=XXXX
export OTHER_APIKEY=XXXX

# Then call uv run stockml-train with command line args to run the program.
$ uv run stockml-train -r run-1
```
## Arguments
`-r, --run-name`: Name of the run to use. If this doesn't match a run name, it will attempt to match the start of a run name, and if only one is found, it will use that run.

`-rf, --run-file`: Path of the file to load the runs from. Default is `runs/model_runs.json`.

`-mf, --model-file`: Path of the file to save/load the model from. 
By default, the model will determine a filename automatically based on architecture parameters, such as `SimpleLSTM-256_72+1.ckpt`.

`-rm, --rebuild-model`: If true then overwrites old model and starts from scratch.
Default behavior is to continue training with the existing checkpoint.

`-d, --device`: Specifies the device to use. Possible values: `cpu`, `cuda`, `mps`.

`-ds, --deepspeed`: Uses Deepspeed to train the model instead of classic PyTorch. Deepspeed can also be used by using `Deepspeed[ModelName]` as the `model_name` in a run.

`-v, --verbosity`: (not implemented)

0: quiet - only run selection, final metrics and trailing predicted prices will be printed.

1: default - the above + announcing each step, and stating basic operations and statistics such as the validation split and number of data rows, and small data previews.

2: diagnostic - the above + model summary, 

## Sample run
```
dqchuang@dqchuang-desktop:~/nas/stockanalysis-ml$ uv run stockml-train -r intra-gmlp
> Reading run file runs/model_runs.json
Copying from run Intra-LSTM-TQQQ
> Running Intra-GMLP-TQQQ

Downloading from AlphaVantage: 100%|█████████████| 8/8 [00:00<00:00, 160.75it/s]
> Model loader parameters:
Using model GatedMLP.
Using device cuda.

> Model config:
StandardConfig(loss='mean_squared_error', optimizer='adam', test_split=0.1, validation_split=0.2, batch_size=64, epochs=4, hidden_layer_size=256, dropout_rate=0.3, seq_len=72, out_seq_len=1)

Total Trainable Params: 1056170
> Loading model from ckpt/GatedMLP-256_72+1.ckpt
> Loading model:
{'epoch': 36, 'loss': 0.0062811562110703005, 'val_loss': 0.005827929126098752}

> Training model GatedMLP.
Epoch 1/4; Total epochs: 37/40
Splitting data at a 0.8 ratio: 25231/6307
395/395 [██████████████████████████████] 00:18 - eta: 00:00, 21.80it/s, loss=0.005767
 99/99  [██████████████████████████████] 00:02 - eta: 00:00, 48.40it/s, val_loss=0.006156
```

## Extra: AMD ROCm installation on Ubuntu 22.04
This worked on AMD Instinct MI25 (`gfx900`)
### Installing AMDGPU
The version I used is 5.4.2, which technically doesn't support the MI25. However, it still seems to work for me, and for a couple other MI25 owners. The last version supporting the MI25 was 4.5.2, which is very old and not supported by PyTorch 2.x.<br>

First of all, make sure the kernel is the right version. The kernel version has long since updated past what ROCm 5.4.2 supports, so you may have to downgrade to 5.15, or whatever kernel version your ROCm supports. See the [ROCm wiki](https://rocm.docs.amd.com/en/docs-5.4.2/release/gpu_os_support.html) for details.

    sudo apt install --install-recommends linux-generic

    # Verify you got the right version - for me, `Ubuntu, with Linux 5.15.0-112-generic` appeared in this list.
    # This shows entries for the GRUB menu.
    sudo grep 'menuentry \|submenu ' /boot/grub/grub.cfg | cut -f2 -d "'"

Now, download and run the driver install script. The --no-dkms option is very important here. For some reason, the GPUs don't get detected (nor does the amdgpu module even attempt to load) when the dkms driver is present. 

    # change 'jammy' to 'focal' for ubuntu 20.x
    wget https://repo.radeon.com/amdgpu-install/5.4.2/ubuntu/jammy/amdgpu-install_5.4.50402-1_all.deb
    sudo apt install ./amdgpu-install_5.4.50402-1_all.deb
    sudo amdgpu-install --usecase=rocm -y --no-dkms
    
### Installing ROCm PyTorch
Run the below to install PyTorch for ROCm 5.4.2. PyTorch 2.0.0 and 2.0.1 support ROCm 5.4.2, so we'll use the latest one. The PyTorch website lists all PyTorch versions, as well as the supported ROCm version (under Wheel->Linux and Windows for each PyTorch version). If you upgrade ROCm (and it works), you should be able to upgrade this as well. https://pytorch.org/get-started/previous-versions/

    pip install torch==2.0.1+rocm5.4.2 torchvision==0.15.2+rocm5.4.2 --index-url https://download.pytorch.org/whl/rocm5.4.2

Then, reboot and in GRUB, select the new kernel version, and run `sudo apt-get autoremove --purge` to remove old kernel versions. This will leave a fallback kernel, so you should install two older kernel versions (keep one as a backup), and remove all of the newer ones. `amdgpu-dkms` will try to compile for every installed kernel version, and will fail if any unsupported kernel versions are present.

### Installing random libraries for DeepSpeed/ROCm not included in Ubuntu
    sudo apt install libstdc++-12-dev libopenmpi-dev libaio-dev rocthrust-dev hipsparse-dev rocblas-dev
### Fix for ROCm torch.compile error:
See [pytorch/pytorch#98707](https://github.com/pytorch/pytorch/issues/98707).

Remove `/tmp/*` files, then add `export ROCM_PATH=/opt/rocm` to .env.
For me, it was sufficient to `rm -rf /tmp/*` without sudo and add the variable.

Fixes:
    
    ...
    SystemError: <built-in function load_binary> returned NULL without setting an exception

    