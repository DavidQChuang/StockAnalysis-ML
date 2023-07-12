
import asyncio
from threading import Thread
from dataclasses import dataclass
import glob
import os
from pathlib import Path
import json

from websockets.server import WebSocketServerProtocol

import sys
from Trainer import load_run, exec_run

RUN_DIR = Path('./runs')

def get_error(message):
    return {
        "type": "error",
        "message": message,
    }
    
async def send_error(ws, message):
    event = {
        "type": "error",
        "message": message,
    }
    await ws.send(json.dumps(event,separators=(',', ':')))
    
def use_target(event):
    # returns event['target'] if target in event, or `false`
    target = ('target' in event) and (event['target'])
    return target, None if target != False else "No target provided."

def use_run_name(event):
    # returns event['target'] if target in event, or `false`
    run_name = ('run_name' in event) and (event['run_name'])
    
    return run_name, None if run_name != False else "No run name provided."
    # Make sure run file has runs
    # if 'runs' in run_file_data:
    #     runs = run_file_data['runs']
        
    #     if run_name in runs:
    #         return run_name, None
    #     else:
    #         return False, "Run not found in run file."
    # else:
    #     return False, "Run file does not contain any runs."

def use_run_path(event):
    path = ('path' in event) and (event['path'])
    
    if path == False:
        return False, "No file path provided."
    
    path = 'runs/' + path
    if os.path.exists(path):
        # must be within run_dir for safety
        if RUN_DIR in Path(path).parents:
            return path, None
        else:
            return False, "File path must be within ./runs."
    else:
        return False, "File path does not exist."
    

class LoggingWrapper:
    def write(self, data):
        if self.__ws != None:
            if data != '\n':
                asyncio.run(self.__ws.send(
                    json.dumps({
                        "type":"log",
                        "message":data
                    }, separators=(",",":"))))
            
        sys.__stdout__.write(data)

    def open(self, ws):
        self.__ws = ws
        sys.stdout = self

    def close(self):
        self.__ws = None
        sys.stdout = sys.__stdout__
        
@dataclass
class WebappState:
    run_files: list[str] = None
    run_file = None
    run_name = None
    run_data = None
    thread = None
    logging_wrapper = LoggingWrapper()
    
def write_state(state):
    with open("webapp-state.json", "w") as file:
        json.dump(state, file)
        
def load_state():
    if os.path.exists("webapp-state.json"):
        with open("webapp-state.json", "r") as file:
            state = json.load(file)
    else:
        state = WebappState()
        
    return state
            
# ###############
# # Command 'read':
# #   Reads something.
# # Args:
# # - target: The type of object to read.
# #   Possible values are 'run_files', 'run_file', 'run'
# def event_read(ws: WebSocketServerProtocol, event, state: WebappState, write_state=False):
#     response = None
    
#     ## Require 'target'
#     event_target, err = use_target(event)
#     if err != None:
#         return get_error(err)
    
#     # If 'lazy=true' is specified, functions will attempt to use 
#     # previous values loaded into state.
#     lazy = False
#     if 'lazy' in event and (str(event['lazy']).lower() == 'true'
#                             or event['lazy'] == True):
#         lazy = True

#     ###############
#     # TARGET: run_files
#     # Read out run files in run dir
#     if event_target == 'run_files':
#         if lazy and state.run_files != None:
#             response = {
#                 'type': 'data',
#                 'data': state.run_files
#             }
#         else:
#             run_files = glob.glob("*.json",root_dir="./runs")
            
#             if write_state:
#                 state.run_files = run_files
                
#             response = {
#                 'type': 'data',
#                 'data': run_files
#             }
    
#     ###############
#     # TARGET: run_file
#     # Read out json contents of run file.
#     # Args:
#     # - path: The relative path of the file to read. Must be in ./runs.
#     elif event_target == 'run_file':
#         if 'path' not in event:
#             lazy = True
            
#         # If no run name is provided, read from state if possible.
#         if lazy == True:
#             # If lazy and no previous run file name, attempt to return loaded state run file data.
#             # Error if no previous run file data exists.
#             if 'path' not in event:
#                 if state.run_file_data == None:
#                     return get_error("Run file path not provided and no run file was previously read or loaded.")
#                 else:
#                     return {
#                         'type': 'data',
#                         'data': state.run_file_data
#                     }
#             # If lazy and run file name provided,
#             # verify the name before attempting to return loaded state run data.
#             # If no previous run file data exists or name doesn't match,
#             # fall back to unlazy loading.
#             else:
#                 if (state.run_name != None 
#                     and state.run_file_path == event['path']
#                     and state.run_file_data != None):
#                         return {
#                             'type': 'data',
#                             'data': state.run_file_data
#                         }
                        
#             # fall back
        
#         ## Require 'path'
#         path, err = use_run_path(event)
#         if err != None:
#             return get_error(err)
        
#         # Attempt to load JSON from path.
#         try:
#             with open(path, "r") as file:
#                 run_file = json.load(file)
            
#             if write_state:
#                 state.run_file_data = run_file
#                 state.run_file_path = path
            
#             response = {
#                 'type': 'data',
#                 'data': run_file
#             }
#         except Exception as e:
#             return get_error(f"Error while reading run file at {path}: {e}.")

#     ###############
#     # TARGET: run
#     # Read out json contents of run. Requires state.run_file_data.
#     elif event_target == 'run':
#         if state.run_file_data == None:
#             return get_error("Run file must be loaded.")
            
#         # If no run name is provided, read from state if possible.
#         if lazy == True:
#             # If lazy and no previous run name, attempt to return loaded state run data.
#             # Error if no previous run data exists.
#             if 'run_name' not in event:
#                 if state.run_data == None:
#                     return get_error("Run name not provided and no run was previously read or loaded.")
#                 else:
#                     return {
#                         'type': 'data',
#                         'data': state.run_data
#                     }
#             # If lazy and run name provided,
#             # verify the name before attempting to return loaded state run data.
#             # If no previous run data exists or name doesn't match,
#             # fall back to unlazy loading.
#             else:
#                 if (state.run_name != None 
#                     and state.run_name == event['run_name']
#                     and state.run_data != None):
#                         return {
#                             'type': 'data',
#                             'data': state.run_data
#                         }
                        
#             # fall back
            
#         ## Require 'run'
#         run_name, err = use_run_name(event)
#         if err != None:
#             return get_error(err)
        
#         state.logging_wrapper.open(ws)
#         run_data = load_run(run_name, state.run_file)
#         state.logging_wrapper.close()
        
#         # Only load run if load event
#         if write_state:
#             state.run_name = run_name
#             state.run_data = run_data
        
#         response = {
#             'type': 'data',
#             'data': run_data
#         }
    
#     # Invalid event target
#     else:
#         return get_error("Invalid event target.")
    
#     # success
#     return response
# def event_load(ws: WebSocketServerProtocol, event, state: WebappState):
#     response = event_read(ws, event, state, write_state=True)
    
#     if response != False and response['type'] != 'error':
#         return {"type":"success"}
#     else:
#         return response

def event_set(ws: WebSocketServerProtocol, event, state: WebappState):
    response = None
    
    ## Require 'target'
    event_target, err = use_target(event)
    if err != None:
        return get_error(err)
    
    # TARGET: run_files
    # Read out run files in run dir
    if event_target == 'run_file':
        print('test')
        

def run_thread(ws: WebSocketServerProtocol, state: WebappState, func, *args):
    if state.thread != None:
        state.thread.join()
        
    def run_func(func, wrapper, ws, *args):
        wrapper.open(ws)
        func(*args)
        wrapper.close()
    
    thread = Thread(target=run_func, args=(func, state.logging_wrapper, ws, *args))
    thread.start()
    
    state.thread = thread

def event_run(ws: WebSocketServerProtocol, event, state: WebappState):
    #if state.run_data == None:
    #    return get_error("Invalid event target.")
    #    return False
    
    run_thread(ws, state, exec_run)
    
    return False
    
import time
def test():
    for x in range(10):
        print(f"x = {x}")
        time.sleep(0.25)