import argparse

import random
import os
import sys
import traceback

import libutil.runs
import libutil.datasets
import libutil.models
import libutil.traders

def main():
    parser = argparse.ArgumentParser(description='Args test')
    parser.add_argument('-r', '--run-name', type=str, dest='run_name', help='Name of the run to use.')
    
    parser.add_argument('-rf', '--run-file', type=str, dest='run_file', help='Path of the file to load the runs from.')
    parser.add_argument('-mf', '--model-file', type=str, dest='model_file', default='', nargs='?', help='Path of the file to save/load the model from.')
    
    parser.add_argument('-rm', '--rebuild-model', dest='rebuild_model', action="store_true", help='Used with --model-file, if true then overwrites old model.')
    parser.add_argument('-v', '--verbosity', type=int, dest='verbosity', help="""0: quiet - only run selection, final metrics and trailing predicted prices will be printed.
1: default - the above + announcing each step, and stating basic operations and statistics such as the validation split and number of data rows, and small data previews.
2: diagnostic - the above + model summary, 
                        """)

    # Parse args
    args = parser.parse_args()
    
    # Verbosity
    verbosity = 1 if args.verbosity is None else args.verbosity
    
    if verbosity < 0 or verbosity > 2:
        print("! Invalid verbosity level. Must be 0-2. ")
        return -1
    
    # Get a run from the run file
    try:
        if args.run_name is None:
            run_data = libutil.runs.from_input("runs/model_runs.json")
        else:
            run_data = libutil.runs.from_file(args.run_file, args.run_name)
        
    except Exception:
        print("! Failed to parse run. Printing exception: ")
        traceback.print_exc()
        return -1
    
    # Start run:
    data = libutil.datasets.from_run(run_data)
    model = libutil.models.from_run(run_data)
    # trader = libutil.traders.from_run(run_data)
    
    model.standard_train(data)
    
    
if __name__ == "__main__":
    main()