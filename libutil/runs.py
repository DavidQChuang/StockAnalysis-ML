import json
import os
import sys

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

def from_file(path: str, run_name: str) -> dict:
    if path is None:
        path="runs/model_runs.json"
        
    if run_name is None:
        raise Exception("'run_name' cannot be None.")
        
    if os.path.exists(path):
        with open(path) as run_file:
            file_json = json.load(run_file)
            
        if 'runs' not in file_json:
            raise Exception("'runs' key must be present in runs file.")
            
        # Get environment variable data
        run_data = {}
        
        if 'env' in file_json:
            for key, value in file_json['env'].items():
                if value not in os.environ:
                    raise Exception(f"Environment variable {value} not found.")
                
                env_value = os.environ[value]
                key_path = key.split('.')
                
                curr = run_data
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
        
        # Get the requested run, merge environment variables into requested without replacing requested values
        if run_name in runs:
            run_data = merge(runs[run_name], run_data)
        else:
            raise Exception("'runs' key must be present in runs file.")
        
        # Merge global run into requested without replacing requested values
        merge(run_data, global_run_data)
        
        return run_data
            
    else:
        raise Exception("Runs file does not exist.")
    
def from_input(path: str) -> dict:
    print("> Selecting run from input. Runs:")
    
    # Check for file
    if os.path.exists(path):
        with open(path) as run_file:
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
            print('Invalid run.')
            continue
        
        do = False
    
    return from_file(path, run_name)
    