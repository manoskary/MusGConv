import musgconv as st
import torch
import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging
# from pytorch_lightning.utilities.seed import seed_everything
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.44)
parser.add_argument('--batch_size', type=int, default=150)
parser.add_argument('--lr', type=float, default=0.0015)
parser.add_argument('--weight_decay', type=float, default=0.0035)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument("--collection", type=str, default="all",
                choices=["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"],  help="Collection to test on.")
parser.add_argument("--predict", action="store_true", help="Obtain Predictions using wandb cloud stored artifact.")
parser.add_argument('--use_jk', action="store_true", help="Use Jumping Knowledge In graph Encoder.")
parser.add_argument('--mtl_norm', default="none", choices=["none", "Rotograd", "NADE", "GradNorm", "Neutral"], help="Which MLT optimization to use.")
parser.add_argument("--include_synth", action="store_true", help="Include synthetic data.")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--use_ckpt", type=str, default=None, help="Use checkpoint for prediction.")
parser.add_argument("--num_tasks", type=int, default=11, choices=[5, 11, 14], help="Number of tasks to train on.")
parser.add_argument("--data_version", type=str, default="v1.0.0", choices=["v1.0.0", "latest"], help="Version of the dataset to use.")
parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs to train for.")
parser.add_argument("--transpose", action="store_true", help="Transpose the training data for Augmentation.")
parser.add_argument("--use_reledge", action="store_true", help="Use relative edge features.")
parser.add_argument("--use_metrical", action="store_true", help="Use metrical features.")
parser.add_argument("--pitch_embedding", type=int, default=None, help="Pitch embedding size to use")
parser.add_argument("--model", type=str, default="MusGConv", choices=["MusGConv", "SageConv", "GatConv", "ResConv"], help="Block Convolution Model to use")
parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging.")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity to use.")
parser.add_argument("--stack_convs", action="store_true", help="Stack convolutions in the model.")
parser.add_argument("--return_edge_emb", action="store_true", help="Input edge embeddings from the previous Encoder layer to the next.")
parser.add_argument("--use_signed_features", action="store_true", help="Use singed instead of absolute edge features in the reledge model. It applies only when use_reledge is True")

# for reproducibility
torch.manual_seed(0)
random.seed(0)
# torch.use_deterministic_algorithms(True)
# seed_everything(seed=0, workers=True)


args = parser.parse_args()
if isinstance(eval(args.gpus), int):
    if eval(args.gpus) >= 0:
        devices = [eval(args.gpus)]
        dev = devices[0]
    else:
        devices = None
        dev = "cpu"
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
    dev = None
n_layers = args.n_layers
n_hidden = args.n_hidden
force_reload = False
num_workers = args.num_workers

first_name = args.mtl_norm if args.mtl_norm != "none" else "Wloss"
name = "{}-{}x{}-lr={}-wd={}-dr={}".format(args.model, n_layers, n_hidden,
                                            args.lr, args.weight_decay, args.dropout)
use_nade = args.mtl_norm == "NADE"
use_rotograd = args.mtl_norm == "Rotograd"
use_gradnorm = args.mtl_norm == "GradNorm"
args.include_synth = True
args.transpose = True
weight_loss = args.mtl_norm not in ["Neutral", "Rotograd", "GradNorm"]
args.model

if args.use_wandb:
    wandb_logger = WandbLogger(
        log_model=True,
        entity=args.wandb_entity,
        project="MusGConv",
        group=f"Roman Numeral Analysis",
        job_type=f"{args.model}-{'wPE' if args.pitch_embedding is not None else 'woPE'}-{'wEF' if args.use_reledge else 'woEF'}-{'wSEF' if args.use_signed_features else 'woSEF'}-{'wEE' if args.return_edge_emb else 'woEE'}",
        # tags=tags,
        name=name)
    wandb_logger.log_hyperparams(args)

datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=args.num_workers, include_synth=args.include_synth, num_tasks=args.num_tasks,
    collection=args.collection, batch_size=args.batch_size, version=args.data_version, include_measures=args.use_metrical, transpose=args.transpose)
model = st.models.chord.MetricalChordPrediction(
    datamodule.features, args.n_hidden, datamodule.tasks, args.n_layers, lr=args.lr, dropout=args.dropout,
    weight_decay=args.weight_decay, use_nade=use_nade, use_jk=args.use_jk, weight_loss=weight_loss,
    use_reledge=args.use_reledge, use_metrical=args.use_metrical, pitch_embedding=args.pitch_embedding,
    conv_block=args.model, stack_convs=args.stack_convs, use_signed_features=args.use_signed_features,
    return_edge_emb=args.return_edge_emb
    )
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="global_step", mode="max")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.02, patience=5, verbose=False, mode="min")
# swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
use_ddp = len(devices) > 1 if isinstance(devices, list) else False
trainer = Trainer(
    max_epochs=args.n_epochs,
    accelerator="auto", devices=devices,
    num_sanity_val_steps=1,
    logger=wandb_logger if args.use_wandb else None,
    # plugins=DDPPlugin(find_unused_parameters=False) if use_ddp else None,
    callbacks=[checkpoint_callback],
    reload_dataloaders_every_n_epochs=5,
    )

# training
trainer.fit(model, datamodule)
# Testing with best model
trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)





