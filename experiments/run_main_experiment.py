import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0", help="GPU to use, -1 for CPU")
parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity to use. Default is your username.")

args = parser.parse_args()
file_path = os.path.dirname(os.path.realpath(__file__))
script_names = ["chord_prediction.py", "voice_separation.py", "cadet.py", "composer_classification.py"]
baseline_model_names = ["SageConv", "ResConv", "SageConv", "SageConv"]


use_wandb = " --use_wandb" if args.use_wandb else ""

for base_model, script_name in zip(script_names, baseline_model_names):
    # Run the previous SOTA architecture
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model {base_model}")

    # Run the MusGConv architecture (still to decide if we include the Edge embedding forwarding
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model MusGConv --use_reledge --pitch_embedding 16")
