import json
import libutil.models
import libutil.traders

import asyncio
from subprocess import PIPE, STDOUT, Popen
from websockets.server import serve, WebSocketServerProtocol
import shutil

from libutil.webapp import WebappState, load_state, event_set, event_run, send_error
    
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
        # try:
        #     event = json.loads(message)
        #     event_type = event['type']
        # except Exception as e:
        #     await send_error(ws, f"Invalid JSON: {e}")
        #     continue
        event = message
        event_type = message.split(' ')[0]

        match event_type:
            case 'set':
                response = event_set(ws, event, STATE)
            
            case 'run':
                response = event_run(ws, event, STATE)
            
            case 'echo':
                response = message
                
            case _:
                response = False
                await send_error(ws, "No event type provided.")
        
        if response != False:
            if not isinstance(response, str):
                try:
                    response = json.dumps(response, separators=(",",":"))
                except Exception as e:
                    await send_error(ws, f"Internal error, response was malformed: {e}")
                    continue
                
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