import torch
import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from musgconv.models.cadence import CadenceClassificationModelLightning
from musgconv.data.datamodules.cadence_dtm import GraphCadenceDataModule
# from pytorch_lightning.plugins import DDPPlugin
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="-1")
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=64)
parser.add_argument('--seed', type=int, default=0, help="Seed for reproducibility")
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument("--num_workers", type=int, default=20)
parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to train for.")
parser.add_argument("--verbose", action="store_true", help="Verbose for dataset loading")
parser.add_argument("--load_from_checkpoint", action="store_true", help="Load model from WANDB checkpoint")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--model", type=str, default="RelEdgeConv", help="Block Convolution Model to use")
parser.add_argument("--reg_loss_weight", type=float, default=0.5, help="Weight of the regularization loss. 0.5 is the default value")
parser.add_argument("--use_jk", action="store_true", help="Use Jumping Knowledge")
parser.add_argument("--tags", type=str, default="", help="Tags to add to the WandB run api")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size related to piece quantity not to graph size.")
parser.add_argument("--use_reledge", action="store_true", help="Use reledge")
parser.add_argument("--use_metrical", action="store_true", help="Use metrical graphs")
parser.add_argument("--pitch_embedding", type=int, default=None, help="Pitch embedding size to use")
parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
parser.add_argument("--aggregation", type=str, default="cat", choices=["cat", "add", "mul"], help="Aggregation method for the edge features in MusGConv.")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity to use.")
parser.add_argument("--stack_convs", action="store_true", help="Stack convolutions of the same type")
parser.add_argument("--heterogeneous", action="store_true", help="Use heterogeneous graphs")
parser.add_argument("--use_all_features", action="store_true", help="Use all features including cadence features and interval vectors otherwise use only 16 features for pitch spelling features and duration.")
parser.add_argument("--return_edge_emb", action="store_true", help="Input edge embeddings from the previous Encoder layer to the next.")
parser.add_argument("--use_signed_features", action="store_true", help="Use singed instead of absolute edge features in the reledge model. It applies only when use_reledge is True")


args = parser.parse_args()
# for reproducibility
seed_everything(seed=args.seed, workers=True)

if args.gpus == "-1":
    devices = None
    use_ddp = False
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
    use_ddp = len(devices) > 1
n_layers = args.n_layers
n_hidden = args.n_hidden
tags = args.tags.split(",")
force_reload = False
num_workers = args.num_workers


name = "{}-{}x{}-lr={}-wd={}-dr={}-rl={}-jk={}".format(args.model,
    n_layers, n_hidden, args.lr,
    args.weight_decay, args.dropout, args.reg_loss_weight, args.use_jk)



datamodule = GraphCadenceDataModule(
    batch_size=args.batch_size, num_workers=num_workers,
    force_reload=force_reload, max_size=1000, verbose=args.verbose, use_all_features=args.use_all_features)
datamodule.setup()
model = CadenceClassificationModelLightning(
    input_features=datamodule.features, output_features=4, use_reledge=args.use_reledge,
    n_layers=n_layers, n_hidden=n_hidden, dropout=args.dropout,
    lr=args.lr, weight_decay=args.weight_decay, metrical=args.use_metrical, use_jk=args.use_jk,
    stack_convs=args.stack_convs, pitch_embedding=args.pitch_embedding, reg_loss_weight=args.reg_loss_weight,
    hetero=args.heterogeneous, conv_block=args.model, use_signed_features=args.use_signed_features,
    return_edge_emb=args.return_edge_emb, use_wandb=args.use_wandb, aggregation=args.aggregation,
)

if args.use_wandb:
    wandb_logger = WandbLogger(
        log_model=True,
        entity=args.wandb_entity,
        project="MusGConv",
        group=f"Cadence Classification",
        job_type=f"{args.model}-{'wPE' if args.pitch_embedding is not None else 'woPE'}-{'wEF' if args.use_reledge else 'woEF'}-{'wSEF' if args.use_signed_features else 'woSEF'}-{'wEE' if args.return_edge_emb else 'woEE'}{'-noCat' if args.aggregation != 'cat' else ''} {'' if args.use_all_features else '-minF'}",
        # tags=tags,
        name=name)
    wandb_logger.log_hyperparams(args)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_f1", mode="max")
trainer = Trainer(
    max_epochs=args.n_epochs, accelerator="auto", devices=devices,
    num_sanity_val_steps=1,
    logger=wandb_logger if args.use_wandb else None,
    # plugins=DDPPlugin(find_unused_parameters=True) if use_ddp else None,
    # replace_sampler_ddp=False,
    reload_dataloaders_every_n_epochs=5,
    callbacks=[checkpoint_callback],
    )


# Training
trainer.fit(model, datamodule)

# Testing with best model
trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)
