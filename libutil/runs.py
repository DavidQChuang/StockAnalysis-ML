import json
import os
import sys

import datasets.indicators as indicators

def except_nokey(dict, key, desc):
    if key not in dict:
        raise Exception(f"'{key}' key must be present in {desc}.")

def merge(a, b, path=None):
    "merges b into a without replacing existing values"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif type(a[key]) is type(b[key]):
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def from_file(run_file: str, run_name: str, **kwargs) -> dict:
    if run_file is None or run_file == "runs/model_runs.json":
        if os.path.exists("runs/model_runs.json"):
            run_file="runs/model_runs.json"
        elif os.path.exists("runs/sample_runs.json"):
            run_file="runs/sample_runs.json"
        else:
            raise Exception("Neither of the default run files exist.")
        
    if run_name is None:
        raise Exception("'run_name' cannot be None.")
        
    print(f"> Reading run file {run_file}")
    if os.path.exists(run_file):
        with open(run_file) as run_file:
            file_json = json.load(run_file)
            
        except_nokey(file_json, 'runs', 'runs file')
            
        # Get environment variable data
        
        if 'env' in file_json:
            for key, value in file_json['env'].items():
                if value not in os.environ:
                    raise Exception(f"Environment variable {value} not found.")
                
                env_value = os.environ[value]
                key_path = key.split('.')
                
                curr = file_json
                for i, part in enumerate(key_path):
                    if i == len(key_path) - 1:
                        curr[part] = env_value
                        break
                        
                    if part not in curr:
                        curr[part] = {}
                        
                    curr = curr[part]
            
        # Get global run data
        if 'global' in file_json:
            global_run_data = file_json['global']
        else:
            global_run_data = {}
        
        # Get runs    
        runs = file_json['runs']
        
        for name in runs.keys():
            if name.lower().startswith(run_name):
                run_name = name
                break
        
        # Get the requested run, merge environment variables into this run without replacing existing values
        except_nokey(runs, run_name, 'runs file; run does not exist')
        
        run_data = runs[run_name]
        
        # Check for 'copy_run' and merge it into this run without replacing existing values
        copied_runs = set([ run_name ])
        while 'copy_run' in run_data:
            copy_run_name = run_data['copy_run']
            del run_data['copy_run']
            
            if copy_run_name not in copied_runs:
                print(f"Copying from run {copy_run_name}")
            
                except_nokey(runs, copy_run_name, 'runs file; run does not exist')
                
                merge(run_data, runs[copy_run_name])
                copied_runs.add(copy_run_name)
        
        # Merge global run into this run without replacing existing values
        merge(run_data, global_run_data)
        
        # Copy model.seq_len and model.out_seq_len into dataset._
        except_nokey(run_data, 'model', 'run')
        except_nokey(run_data, 'dataset', 'run')
        except_nokey(run_data["model"], 'seq_len', 'run.model')
        except_nokey(run_data["model"], 'out_seq_len', 'run.model')
        run_data["dataset"]['seq_len'] = run_data["model"]["seq_len"]
        run_data["dataset"]['out_seq_len'] = run_data["model"]["out_seq_len"]
        
        # Copy dataset.columns into model.columns
        if 'columns' not in run_data['dataset']:
            run_data["dataset"]['columns'] = [{ "name": 'close', "is_scaled": True}]
        run_data["model"]['columns'] = run_data["dataset"]['columns']
            
        # Create dataset.columns from dataset.indicators
        if 'indicators' in run_data['dataset']:
            run_data['model']['indicators'] = run_data['dataset']['indicators']
            
            for indicator in run_data['dataset']['indicators']:
                if 'is_input' in indicator and indicator['is_input']:
                    if 'name' not in indicator:
                        indicator['name'] = indicators.get_indicator_name_json(indicator)
                    ind_name = indicator['name']
                    
                    # model.columns updates with dataset.columns
                    run_data['dataset']['columns'].append({
                        "name": ind_name,
                        "from_indicator": True,
                        "is_scaled": 'is_scaled' in indicator and indicator['is_scaled']
                    })
        
        print(f"> Running {run_name}")
        print()
        
        return run_data
            
    else:
        raise Exception("Runs file does not exist.")
    
def from_input(run_file: str, **kwargs) -> dict:
    print("> Selecting run from input. Runs:")
    
    # Check for file
    if os.path.exists(run_file):
        with open(run_file) as run_file:
            file_json = json.load(run_file)
            
        if 'runs' not in file_json:
            raise Exception("'runs' key must be present in runs file.")
        
        # Get run names
        run_names = file_json['runs'].keys()
        
        for run in run_names:
            print(run)
            
    else:
        raise Exception("Runs file does not exist.")
    
    # Select run from input
    do = True
    while do:
        run_name = input("> Select run (enter to exit): ")
        
        if run_name == "":
            break
        
        for name in run_names:
            if name.lower().startswith(run_name):
                run_name = name
                break
        
        if not run_name in run_names:
            print('! Invalid run.')
            continue
        
        do = False
    
    return from_file(run_file, run_name)
    