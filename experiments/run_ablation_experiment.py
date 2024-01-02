import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0", help="GPU to use, -1 for CPU")
parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity to use. Default is your username.")
parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")

args = parser.parse_args()
file_path = os.path.dirname(os.path.realpath(__file__))


args = parser.parse_args()
file_path = os.path.dirname(os.path.realpath(__file__))
script_names = ["chord_prediction.py", "voice_separation.py", "cadet.py", "composer_classification.py"]

use_wandb = " --use_wandb" if args.use_wandb else ""

for script_name in script_names:
    # The MusGConv model with pitch embedding (16). This is the model we use in the paper.
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model MusGConv --use_reledge --pitch_embedding 16 --seed {args.seed}")

    # The MusGConv model with edge embedding forwarding
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model MusGConv --use_reledge --pitch_embedding 16 --return_edge_emb --seed {args.seed}")

    # The MusGConv model with signed edge features
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model MusGConv --use_reledge --pitch_embedding 16 --use_signed_features --seed {args.seed}")

    # The MusGConv model without pitch embedding
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model MusGConv --use_reledge --seed {args.seed}")

    # The MusGConv model without edge features (i.e. reledge set to False)
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model MusGConv --seed {args.seed}")

    # The MusGConv model with multiplication aggregation (instead of concatenation) of edge features
    os.system(f"python {file_path}/{script_name} --gpus {args.gpus}{use_wandb} --wandb_entity {args.wandb_entity} --model MusGConv --use_reledge --aggregation mul --seed {args.seed}")