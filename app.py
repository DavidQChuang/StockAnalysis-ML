import json
import libutil.models
import libutil.traders

import asyncio
from subprocess import PIPE, STDOUT, Popen
from websockets.server import serve, WebSocketServerProtocol
import shutil

from libutil.webapp import get_success, get_error
from libutil.webapp import WebappState, load_state, save_state, event_set, event_load, event_run
    
async def run_test(ws):
    # NOTE: the `-u` is required for printing unbuffered (I believe) to stdout
    # args = [shutil.which("python3"), "-u", "print_text.py"]
    args = [shutil.which("ping"), "-c", "10", "<HOSTNAME>"]
    with Popen(
        args, stdout=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True
    ) as proc:
        for line in proc.stdout:
            await ws.send(line)

STATE = load_state()

async def echo(ws: WebSocketServerProtocol):
    async for message in ws:
        event = message.split(' ')
        event_type = event[0]

        match event_type:
            case 'state':
                if len(event) == 1:
                    response = json.dumps(vars(STATE))
                elif len(event) == 2:
                    if event[1] == 'save':
                        try:
                            save_state(STATE)
                            response = get_success()
                        except Exception as e:
                            response = get_error(f"Failed to save state: {e}")
            
            case 'set':
                response = event_set(ws, event, STATE)
                
            case 'load':
                response = event_load(ws, event, STATE)
            
            case 'run':
                response = event_run(ws, event, STATE)
            
            case 'echo':
                response = message
                
            case _:
                response = get_error("Invalid command.")
        
        if response != False:
            if not isinstance(response, str):
                try:
                    response = json.dumps(response, separators=(",",":"))
                except Exception as e:
                    response = get_error(f"Internal error, response was malformed: {e}")
                
            await ws.send(response)
            
import time
async def test():
    for x in range(10):
        print(f"x = {x}")
        time.sleep(0.25)
        
async def main():
    async with serve(echo, "localhost", 8701):
        await asyncio.Future()  # run forever


if __name__ == '__main__':
    asyncio.run(main())
    # socket_.run(app, debug=True)