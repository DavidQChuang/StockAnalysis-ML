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
    parser.add_argument('-r', '--run-name', type=str, dest='run_name',
                        help='Name of the run to use.')
    
    parser.add_argument('-rf', '--run-file', type=str, dest='run_file',
                        default="runs/model_runs.json",
                        help='Path of the file to load the runs from.')
    
    parser.add_argument('-mf', '--model-file', type=str, dest='model_file',
                        help='Overrides the default path of the file to save/load the model from.'
                        +    'By default, the model decides a filename based on its parameters.')
    
    parser.add_argument('-tf', '--trader-file', type=str, dest='trader_file',
                        help='Overrides the default path of the file to save/load the model from.'
                        +    'By default, the model decides a filename based on its parameters.')
    
    parser.add_argument('-rm', '--rebuild-model', dest='rebuild_model', action="store_true",
                        help='Used with --model-file, if true then overwrites old model.')
    
    parser.add_argument('-rt', '--rebuild-trader', dest='rebuild_trader', action="store_true",
                        help='Used with --trader-file, if true then overwrites old trader.')
    
    parser.add_argument('-v', '--verbosity', type=int, dest='verbosity',
                        help="""0: quiet - only run selection, final metrics and trailing predicted prices will be printed.
1: default - the above + announcing each step, and stating basic operations and statistics such as the validation split and number of data rows, and small data previews.
2: diagnostic - the above + model summary, 
                        """)

    parser.add_argument('-d', '--device', type=str, dest='device',
                        help='Device to run the model on (cuda, cpu).')

    parser.add_argument('-ds', '--deepspeed', dest='use_deepspeed', action="store_true",
                        help='Enables deepspeed.')

    # Parse args
    args = parser.parse_args()
    args_dict = vars(args)
        
    # Verbosity
    verbosity = 1 if args.verbosity is None else args.verbosity
    
    if verbosity < 0 or verbosity > 2:
        print("! Invalid verbosity level. Must be 0-2. ")
        return -1
    
    # Get a run from the run file
    try:
        if args.run_name is None:
            run_data = libutil.runs.from_input(**args_dict)
        else:
            run_data = libutil.runs.from_file(**args_dict)
        
    except Exception:
        print("! Failed to parse run. Printing exception: ")
        traceback.print_exc()
        return -1
    
    # Start run:
    dataset = libutil.datasets.from_run(run_data, **args_dict)
    model = libutil.models.from_run(run_data, **args_dict)
    trader = libutil.traders.from_run(run_data, **args_dict)
    
    if not os.path.isdir("ckpt"):
        os.mkdir("ckpt")
    
    model_file = load_thing(model, args.model_file, args.rebuild_model)
    model.standard_train(dataset)
    
    if not model.conf.epochs == 0:
        print(f"> Saving model to {model_file}")
        model.save(model_file)
        
    if trader is not None:
        trader_file = load_thing(trader, args.trader_file, args.rebuild_trader)
        trader.standard_train(model, dataset)
    
        if not trader.conf.episodes == 0:
            print(f"> Saving trader to {trader_file}")
            trader.save(trader_file)
            
def load_thing(model, model_file, rebuild_model):
    model_file = f"ckpt/{model.get_filename()}" if model_file == None else model_file
    
    if not rebuild_model:
        if os.path.exists(model_file):
            print(f"> Loading model from {model_file}")
            model.load(model_file)
        else: 
            print(f"> No model to load at {model_file}")
    else:
        if os.path.exists(model_file):
            print(f"> Overwriting model at {model_file}, will not be loaded")
        else:
            print(f"> No model at {model_file}")
    
    return model_file
    
if __name__ == "__main__":
    main()