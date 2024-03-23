import argparse
from datetime import datetime, timedelta
import numpy as np
from pytz import timezone

import random
import os
import sys
import traceback

import libutil.runs
import libutil.datasets
import libutil.models
import libutil.traders
from libutil.visualizer import VApp, VWorker, run_app, visualize_module

from models.GatedMLP import GatedMLP

def main_cmd():
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

    parser.add_argument('-vs', '--visualizer', dest='use_visualizer', action="store_true",
                        help='Enables visualizer.')

    parser.add_argument('-ei', '--eval-inference', dest='eval_inference_count', type=int,
                        help='Enables evaluation mode for inference. Gets the latest data and then runs the given number of inferences and prints them.')

    # parser.add_argument('-et', '--eval-trader', dest='eval_trader_count', type=int,
    #                     help="""Enables evaluation mode for trading. Gets the latest data and then runs the given number of inferences, then runs the trader and prints
    #                     the inferences and the trader's actions.""")

    # Parse args
    args = parser.parse_args()
    
    # If using visualizer, enter the Qt event loop and let it run the program.
    if args.use_visualizer == True:
        run_app(lambda worker, app: main(args, worker, app))
    else:
        main(args)

def main(args: argparse.Namespace, worker: VWorker|None = None, app: VApp|None = None):
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
    
    if worker != None:
        worker.sig_dataset.emit(dataset)
    
    # Run mode
    if args.eval_inference_count == None: # and args.eval_trader_count == None:
        trader = libutil.traders.from_run(run_data, **args_dict)
        
        if not os.path.isdir("ckpt"):
            os.mkdir("ckpt")
        
        if not model.conf.epochs == 0:
            model_file = load_thing(model, args.model_file, args.rebuild_model)
            model.standard_train(dataset,
                iter_callback=None if worker == None else worker.iter_callback(),
                data_callback=None if worker == None else worker.data_callback()) # type: ignore
            
            print(f"> Saving model to {model_file}")
            model.save(model_file)
        else:
            print("> No model epochs, skipping.")
            
        if trader is not None:
            if not trader.conf.episodes == 0:
                trader_file = load_thing(trader, args.trader_file, args.rebuild_trader)
                trader.standard_train(model, dataset)
        
                print(f"> Saving trader to {trader_file}")
                trader.save(trader_file)
            else:
                print("> No trader episodes, skipping.")
    else: # and args.eval_trader_count != None:
        # Load model
        model_file = load_thing(model, args.model_file, args.rebuild_model)
        
        tz = timezone('EST')
        strftime_format = "%A, %D %X %Z"

        # Print current date
        print()
        print("# [current_time] Current date: ", datetime.now(tz).strftime(strftime_format))
        
        # Get timestep of dataset
        dt_cols = dataset.df.select_dtypes(include=[np.datetime64]) # type: ignore ; this is a valid argument
        timestep: timedelta
        
        dataset.df.to_csv("csv/test.csv")
        
        if verbosity >= 2:
            print(f"@ datetime datatypes ('timestamp' included ? {'timestamp' in dt_cols.columns}): ", dt_cols)
        if 'timestamp' in dt_cols:
            dt_delta = dt_cols['timestamp'].diff().iloc[1:]
            if verbosity >= 2:
                print("@ dt_delta: ", dt_delta)
                # print("@ idxmin-1: ", dt_delta.idxmin(), dataset.df.loc[dt_delta.idxmin()-1, :])
                print("@ idxmin: ", dt_delta.idxmin(), dataset.df.loc[dt_delta.idxmin(), :])
            timestep = dt_delta.min()
        else:
            raise KeyError("No 'timestamp' column with np.datetime64 dtype. Datasets should include such a column, or it is not being loaded properly.")
            
        print(f"# [timestep] Detected timestep ({timestep.days} days {timestep.seconds//3600:02d}:{timestep.seconds//60 % 60:02d}:{timestep.seconds%60:02d}).")
            
        if timestep <= timedelta(minutes=0):
            print("!! timestep: ", timestep)
            raise ValueError("Detected timestep was non-positive. The dataset is in the wrong order or has duplicate elements.")
        
        # Make and print inferences
        curr_date: datetime = dataset.df['timestamp'].iloc[-1] # when the inference occurs
        
        # for i in range(args.eval_inference_count):
        
        print(dataset.df)
        
        for i in range(-args.eval_inference_count, 1):
            curr_input = dataset.get(i)['X']  # type: ignore ; np array with input data
            output = model.infer(curr_input).item()
            
            if i != 0:
                print(f"# [realvalue] [step={i}], Date: [date={curr_date.strftime(strftime_format)}], Value($): [val={dataset.df['close'].iloc[i]}]")
            print(f"# [inference] [step={i}], Date: [date={curr_date.strftime(strftime_format)}], Value($): [val={output:.2f}]")
            curr_date += timestep
            
            
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
    main_cmd()