
import asyncio
import inspect
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
    
# Wrapper for logging stdout
class LogRedirect:
    def __init__(self, ws, passthrough):
        self.__ws = ws
        self.__passthrough = passthrough
    
    def write(self, data):
        if data != '\n':
            asyncio.run(self.__ws.send(
                json.dumps({
                    "type":"log",
                    "message":data
                }, separators=(",",":"))))
            
        if self.__passthrough != None:
            self.__passthrough.write(data)

class LoggingWrapper:
    def __init__(self, ws):
        self.__ws = ws
    
    def __enter__(self):
        self.open()
    
    def __exit__(self, type, value, traceback):
        self.close()
    
    def open(self):
        sys.stdout = LogRedirect(self.__ws, sys.__stdout__)
        sys.stderr = LogRedirect(self.__ws, sys.__stderr__)

    def close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
# Stores webapp state
@dataclass
class WebappState:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    run_files: list[str] = None
    run_file: str = "runs/model_runs.json"
    run_name: str = None
    run_data: dict = None
    thread = None
    
# Read/write state to file
def save_state(state):
    with open("webapp-state.json", "w") as file:
        json.dump(vars(state), file, indent=4)
        
def load_state():
    if os.path.exists("webapp-state.json"):
        try:
            with open("webapp-state.json", "r") as file:
                data = json.load(file)
                state = WebappState.from_dict(data)
        except Exception as e:
            print(f"Failed to load state from file: {e}")
            state = WebappState()
    else:
        state = WebappState()
        
    return state

def get_success():
    return {"type":"success"}

# 
def get_error(message):
    return {
        "type": "error",
        "message": message,
    }
            
def event_set(ws: WebSocketServerProtocol, event, state: WebappState):
    response = None
    
    ## Require 'target'
    if len(event) != 3:
        return get_error("Invalid number of arguments.")
    
    # set {event_target} {event_value}
    event_target = event[1]
    event_value = event[2]
    
    # Set run file to load runs from
    if event_target == 'run_file':
        state.run_file = event_value
        response = get_success()
        
    # Set run name
    elif event_target == 'run_name':
        state.run_name = event_value
        response = get_success()
        
    return response

def event_load(ws: WebSocketServerProtocol, event, state: WebappState):
    response = None
    
    ## Require 'target'
    if len(event) != 2:
        return get_error("Invalid number of arguments.")
    
    # load {event_target}
    event_target = event[1]
    
    # Set run file to load runs from
    if event_target == 'run':
        if state.run_file == None:
            return get_error("Run file not set. Use 'set run_file [value]' to set.")
        if state.run_name == None:
            return get_error("Run name not set. Use 'set run_name [value]' to set.")
        
        try:
            def _load():
                with LoggingWrapper(ws):
                    state.run_data = load_run(state.run_name, state.run_file)
            
            thread = Thread(target = _load)
            thread.start()
            thread.join()
            
        except Exception as e:
            return get_error(f"Failed to load run: {e}")
            
        response = get_success()
        
    return response

def event_run(ws: WebSocketServerProtocol, event, state: WebappState):
    if state.run_data == None:
       return get_error("Run data not loaded. Set run_file and run_name then run 'load run'.")
    
    run_thread(ws, state, exec_run, state.run_data)
    
    return False
    

def run_thread(ws: WebSocketServerProtocol, state: WebappState, func, *args):
    if state.thread != None:
        state.thread.join()
        
    def run_func(func, ws, *args):
        with LoggingWrapper(ws):
            func(*args)
    
    thread = Thread(target=run_func, args=(func, ws, *args))
    thread.start()
    
    state.thread = thread
    
import time
def test():
    for x in range(10):
        print(f"x = {x}")
        time.sleep(0.25)