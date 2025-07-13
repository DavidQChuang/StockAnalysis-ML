import json
import os
from typing import Tuple

from networkx import is_path

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

def get_path_in(path, dictionary):
    parts = path.split('.')
    
    for part in parts:
        if type(dictionary) is dict and part in dictionary:
            dictionary = dictionary[part]
        else:
            return False
            
    return dictionary

def from_file(run_file: str, run_name: str, **kwargs) -> Tuple[dict, str]:
    print(f"[###] > Reading runs from runfile (default: runs/model_runs.json or runs/sample_runs.json).")
    
    if run_file is None or run_file == "runs/model_runs.json":
        if os.path.exists("runs/model_runs.json"):
            run_file="runs/model_runs.json"
        elif os.path.exists("runs/sample_runs.json"):
            run_file="runs/sample_runs.json"
        else:
            raise Exception("Neither of the default run files exist.")
        
    if run_name is None:
        raise Exception("'run_name' cannot be None.")
        
    print(f"├[1/3] > Reading run file {run_file}")
    if os.path.exists(run_file):
        with open(run_file) as file:
            file_json = json.load(file)
            
        except_nokey(file_json, 'runs', 'runs file')
            
        # Get environment variable data
        if 'env' in file_json:
            for key, value in file_json['env'].items():
                if value not in os.environ:
                    if value in kwargs:
                        env_value = kwargs[value]
                    else:
                        raise Exception(f"Environment variable {value} not found.")
                else: 
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
        
        # Find full run name.
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
                print(f"├[2/3] > Copying from run {copy_run_name}")
            
                except_nokey(runs, copy_run_name, 'runs file; run does not exist')
                
                merge(run_data, runs[copy_run_name])
                copied_runs.add(copy_run_name)
                
                
        # Merge global * source into this run's source without replacing existing values
        global_source_json = get_path_in('dataset.sources.*', global_run_data)
        if global_source_json:
            run_source_data = get_path_in('dataset.sources', run_data)
            if run_source_data:
                for source_name, source_json in run_source_data.items():
                    # Overwrite the global source_json with the specific source_json
                    merge(source_json, global_source_json)
                    
                    run_source_data[source_name] = source_json
        
            del global_run_data['dataset']['sources']['*']
        
        # Merge global run into this run without replacing existing values
        merge(run_data, global_run_data)
        
        print(f"└[3/3] > Loaded {run_name}")
        print()
        
        return run_data, run_name
            
    else:
        raise Exception("Runs file does not exist.")
    
def from_input(run_file: str, **kwargs) -> Tuple[dict, str]:
    print(f"[###] > Reading run from runfile (default: runs/model_runs.json or runs/sample_runs.json).")
    
    if run_file is None or run_file == "runs/model_runs.json":
        if os.path.exists("runs/model_runs.json"):
            run_file="runs/model_runs.json"
        elif os.path.exists("runs/sample_runs.json"):
            run_file="runs/sample_runs.json"
        else:
            raise Exception("Neither of the default run files exist.")
        
    # Check for file
    if os.path.exists(run_file):
        with open(run_file) as file:
            file_json = json.load(file)
            
        if 'runs' not in file_json:
            raise Exception("'runs' key must be present in runs file.")
        
        # Get run names
        run_names = file_json['runs'].keys()
        
        for run in run_names:
            print(run)
            
    else:
        raise Exception("Runs file does not exist.")
    
    # Select run from input
    while True:
        print("> Selecting run from stdin.")
        run_name = input("> Provide run name. Enter to exit: ")
        
        if run_name == "":
            print("! Aborting.")
            return {}, None
        
        for name in run_names:
            if name.lower().startswith(run_name):
                run_name = name
                break
        
        if not run_name in run_names:
            print('! Invalid run.')
            continue
        
        break
    
    return from_file(run_file, run_name)
    