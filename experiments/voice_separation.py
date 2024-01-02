import os.path
import torch
import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning import Trainer, seed_everything
from musgconv.models.vocsep import MetricalVoiceLinkPredictionModel
from musgconv.data.datamodules.mix_vs import GraphMixVSDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--collection', type=str, default="split")
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--seed', type=int, default=0, help="Seed for reproducibility")
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument("--pot_edges_max_dist", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=20)
parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--load_from_checkpoint", action="store_true", help="Load model from WANDB checkpoint")
parser.add_argument("--linear_assignment", action="store_true", help="Use linear assignment Hungarian algorithm for val and test predictions.")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--model", type=str, default="RelEdgeConv", help="Block Convolution Model to use")
parser.add_argument("--reg_loss_weight", type=str, default="auto", help="Weight of the regularization loss. If 'auto', it augments every epoch end.", choices=["auto", "none", "fixed"])
parser.add_argument("--use_jk", action="store_true", help="Use Jumping Knowledge")
parser.add_argument("--tags", type=str, default="", help="Tags to add to the WandB run api")
parser.add_argument("--homogeneous", action="store_true", help="Use homogeneous graphs")
parser.add_argument("--reg_loss_type", type=str, default="la", help="Use different regularization loss")
parser.add_argument("--batch_size", type=int, default=28, help="Batch size")
parser.add_argument("--use_reledge", action="store_true", help="Use reledge")
parser.add_argument("--use_metrical", action="store_true", help="Use metrical graphs")
parser.add_argument("--pitch_embedding", type=int, default=None, help="Pitch embedding size to use")
parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity to use.")
parser.add_argument("--aggregation", type=str, default="cat", choices=["cat", "add", "mul"], help="Aggregation method for the edge features in MusGConv.")
parser.add_argument("--stack_convs", action="store_true", help="Stack convolutions")
parser.add_argument("--return_edge_emb", action="store_true", help="Input edge embeddings from the previous Encoder layer to the next.")
parser.add_argument("--use_signed_features", action="store_true", help="Use singed instead of absolute edge features in the reledge model. It applies only when use_reledge is True")

args = parser.parse_args()
# for reproducibility
seed_everything(seed=args.seed, workers=True)


if args.gpus == "-1":
    devices = 1
    use_ddp = False
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
    use_ddp = len(devices) > 1
rev_edges = "new_type"
collections = args.collection.split(",") if args.collection != "split" else args.collection
n_layers = args.n_layers
n_hidden = args.n_hidden
linear_assignment = args.linear_assignment
pot_edges_max_dist = args.pot_edges_max_dist
tags = args.tags.split(",")
force_reload = False
num_workers = args.num_workers


name = "{}-{}x{}-{}-lr={}-wd={}-dr={}-rl={}-jk={}".format(args.model,
    n_layers, n_hidden, "wLN" if args.linear_assignment else "woLN", args.lr,
    args.weight_decay, args.dropout, args.reg_loss_weight, args.use_jk)

if args.use_wandb:
    wandb_logger = WandbLogger(
        log_model=True,
        entity=args.wandb_entity,
        project="MusGConv",
        group=f"Voice Separation - {collections if isinstance(collections, str) else collections[0]}",
        job_type=f"{args.model}-{'wPE' if args.pitch_embedding is not None else 'woPE'}-{'wEF' if args.use_reledge else 'woEF'}-{'wSEF' if args.use_signed_features else 'woSEF'}-{'wEE' if args.return_edge_emb else 'woEE'}{'-noCat' if args.aggregation != 'cat' else ''}",
        # tags=tags,
        name=name)
    wandb_logger.log_hyperparams(args)

datamodule = GraphMixVSDataModule(
    batch_size=args.batch_size, num_workers=num_workers,
    force_reload=force_reload, test_collections=collections,
    pot_edges_max_dist=pot_edges_max_dist, include_measures=args.use_metrical)
datamodule.setup()
if args.homogeneous:
    raise NotImplementedError("Homogeneous Metrical graphs are not implemented yet.")
else:
    model = MetricalVoiceLinkPredictionModel(
        datamodule.features, n_hidden,
        n_layers=n_layers, lr=args.lr, dropout=args.dropout,
        weight_decay=args.weight_decay, linear_assignment=linear_assignment,
        block=args.model, jk=args.use_jk, reg_loss_weight=args.reg_loss_weight,
        reg_loss_type=args.reg_loss_type, use_reledge=args.use_reledge, use_metrical=args.use_metrical,
        pitch_embedding=args.pitch_embedding, stack_convs=args.stack_convs, use_signed_features=args.use_signed_features,
        return_edge_emb=args.return_edge_emb, aggregation=args.aggregation,)

print("Only monophonic:", model.linear_assignment)
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_fscore", mode="max")
lr_monitor = LearningRateMonitor(logging_interval='step')
# swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
trainer = Trainer(
    max_epochs=args.n_epochs, accelerator="auto", devices=devices,
    num_sanity_val_steps=1,
    logger=wandb_logger if args.use_wandb else None,
    # plugins=DDPPlugin(find_unused_parameters=True) if use_ddp else None,
    # replace_sampler_ddp=False,
    reload_dataloaders_every_n_epochs=5,
    callbacks=[checkpoint_callback, lr_monitor],
    )

# Find Batch Size
# tuner = Tuner(trainer)
# tuner.scale_batch_size(model, datamodule, mode="power")

# Training
trainer.fit(model, datamodule)

# Testing with best model
trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)
