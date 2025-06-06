# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import dill as pickle
import sys, os, torch, math, time, warnings
import torch_pruning as tp
import matplotlib

from ultralytics.nn.extra_modules.transformer import DPRA
matplotlib.use('AGG')
import matplotlib.pylab as plt
import torch.nn as nn
import numpy as np
import subprocess
from datetime import datetime, timedelta
from torch import optim
from thop import clever_format, profile
from functools import partial
from torch import distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from .CSPLayer_transfer import replace_CSPLayer_with_CSPLayer_v2
from torch.utils.tensorboard import SummaryWriter

from copy import copy, deepcopy
from pathlib import Path

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, TQDM, __version__, callbacks, clean_url, colorstr, emojis,
                               yaml_save)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer, torch_distributed_zero_first, get_num_params)
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from .train import RTDETRTrainer
from .val import RTDETRDataset, RTDETRValidator

from ultralytics.nn.modules import RepConv, AIFI, RTDETRDecoder, HGStem, HGBlock, LightConv
from ultralytics.nn.extra_modules.block import DEFBlock, DEFModule, RepConvN, SPDConv, SMEModule 
from ultralytics.nn.extra_modules.prune_module import DEFBlock_v2
from ultralytics.nn.extra_modules.MyPruner import RepConvPruner, RepConvNPruner

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_pruner(opt, model, example_inputs):
    sparsity_learning = False
    if opt.prune_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "l1": 
        # https://arxiv.org/abs/1608.08710
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "lamp":
        # https://arxiv.org/abs/2010.07611
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "slim":
        # https://arxiv.org/abs/1708.06519
        # sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_slim":
        # https://tibshirani.su.domains/ftp/sparse-grlasso.pdf
        # sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning, group_lasso=True)
    elif opt.prune_method == "group_norm":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_sl":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        # sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "growing_reg":
        # https://arxiv.org/abs/2012.09243
        # sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=opt.reg, delta_reg=opt.delta_reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_hessian":
        # https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
        imp = tp.importance.HessianImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_taylor":
        # https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
        imp = tp.importance.TaylorImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    customized_pruners = {}
    round_to = 8
        
    customized_pruners[RepConv] = RepConvPruner()
    for k, m in model.named_modules():
        if isinstance(m, DPRA):
            ignored_layers.append(m)
        if isinstance(m, RTDETRDecoder):
            ignored_layers.append(m)
        if isinstance(m, SPDConv):
            ignored_layers.append(m)
        if isinstance(m, SMEModule):
            ignored_layers.append(m)
        if isinstance(m, DEFModule):
            ignored_layers.append(m)
    ignored_layers.append(model.model[11])
    ignored_layers.append(model.model[16])
    
    print(ignored_layers)
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=opt.iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=opt.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        customized_pruners=customized_pruners,
        round_to=round_to,
        root_module_types=[torch.nn.Conv2d, torch.nn.Linear, RepConv, RepConvN]
    )
    return sparsity_learning, imp, pruner

linear_trans = lambda epoch, epochs, reg, reg_ratio: (1 - epoch / (epochs - 1)) * (reg - reg_ratio) + reg_ratio

class RTDETRCompressor(BaseTrainer):
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        check_resume (method): Method to check if training should be resumed from a saved checkpoint.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            (self.save_dir / 'visual').mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.sl_epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split('.')[-1] in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(','))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning("WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                               "default 'batch=16'")
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device('cuda', RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # set to enforce timeout
        dist.init_process_group(
            'nccl' if dist.is_nccl_available() else 'gloo',
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size)

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            # elif not v.requires_grad:
            #     LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
            #                 'See ultralytics.engine.trainer for customization of frozen layers.')
            #     v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(False).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.args.nbs = self.batch_size
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')



    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)),
            'ema': deepcopy(self.ema.ema),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last and best
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if self.best_sl[f'{self.sparsity_ratio:.3f}'] == self.fitness:
            torch.save({'model': deepcopy(de_parallel(self.model)),
                        'ema': deepcopy(self.ema.ema),}, self.wdir / 'best_sl_{:.3f}.pt'.format(self.sparsity_ratio))
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')

    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        return data['train'], data.get('val') or data.get('test')

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients (Align from PaddlePaddle)
        # self.scaler.step(self.optimizer)
        self.optimizer.step()
        # self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        bs = len(batch['img'])
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = RTDETRDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """
        Returns a DetectionValidator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build dataset."""
        return RTDETRDataset(img_path=img_path,
                             imgsz=self.args.imgsz,
                             batch_size=batch,
                             augment=mode == 'train',
                             hyp=self.args,
                             rect=False,
                             cache=self.args.cache or None,
                             prefix=colorstr(f'{mode}: '),
                             data=self.data)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch + 1] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {'data': data, 'timestamp': time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args['data']).exists():
                    ckpt_args['data'] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = str(last)  # reinstate model
                for k in 'imgsz', 'batch':  # allow arg updates to reduce memory on resume if crashed due to CUDA OOM
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            LOGGER.info(f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                        f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                        f"determining best 'optimizer', 'lr0' and 'momentum' automatically... ")
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer

    def remove_all_hooks(self, model):
        """
        移除模型中所有层的 forward hooks, forward pre-hooks 和 backward hooks
        """
        for module in model.modules():  # 遍历模型的所有子模块
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.clear()
            if hasattr(module, "_backward_hooks"):
                module._backward_hooks.clear()
    
    def validate_prune(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        self.ema.ema = deepcopy(self.model)
        self.remove_all_hooks(self.ema.ema)
        metrice = self.validator(self)
        self.ema.ema = None
        torch.cuda.empty_cache()
        return metrice
    
    def model_prune(self, imp, prune, example_inputs):
        N_batchs = 10

        base_model = deepcopy(self.model)
        with HiddenPrints():
            try:
                ori_flops, ori_params = tp.utils.count_ops_and_params(base_model, example_inputs)
            except:
                ori_flops, _ = profile(base_model, [example_inputs])
                ori_params = get_num_params(base_model)
        ori_flops = ori_flops * 2.0
        ori_flops_f, ori_params_f = clever_format([ori_flops, ori_params], "%.3f")
        ori_result = self.validate_prune()
        ori_map50, ori_map = ori_result['metrics/mAP50(B)'], ori_result['metrics/mAP50-95(B)']
        iter_idx, prune_flops = 0, ori_flops
        speed_up = 1.0
        LOGGER.info('begin pruning...')
        while speed_up < self.args.speed_up:
            self.model.train()
            if isinstance(imp, tp.importance.HessianImportance):
                self.remove_all_hooks(self.model)
                for k, batch in enumerate(self.train_loader):
                    if k >= N_batchs: break
                    batch = self.preprocess_batch(batch)
                    # compute loss for each sample
                    loss = self.model(batch)[0]
                    imp.zero_grad() # clear accumulated gradients
                    self.model.zero_grad() # clear gradients
                    loss.backward(retain_graph=True) # simgle-sample gradient
                    imp.accumulate_grad(self.model) # accumulate g^2
            elif isinstance(imp, tp.importance.TaylorImportance):
                self.remove_all_hooks(self.model)
                for k, batch in enumerate(self.train_loader):
                    if k >= N_batchs: break
                    batch = self.preprocess_batch(batch)
                    loss = self.model(batch)[0]
                    loss.backward()
            
            iter_idx += 1
            prune.step(interactive=False)
            prune_result = self.validate_prune()
            prune_map50, prune_map = prune_result['metrics/mAP50(B)'], prune_result['metrics/mAP50-95(B)']
            with HiddenPrints():
                try:
                    prune_flops, prune_params = tp.utils.count_ops_and_params(self.model, example_inputs)
                except:
                    prune_flops, _ = profile(self.model, [example_inputs])
                    prune_params = get_num_params(self.model)
            prune_flops = prune_flops * 2.0
            prune_flops_f, prune_params_f = clever_format([prune_flops, prune_params], "%.3f")
            speed_up = ori_flops / prune_flops # ori_model_GFLOPs / prune_model_GFLOPs
            LOGGER.info(f'pruning... iter:{iter_idx} ori model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%) map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f}) map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f}) Speed Up:{ori_flops / prune_flops:.2f}')
            
            if prune.current_step == prune.iterative_steps:
                break
        
        if isinstance(imp, tp.importance.HessianImportance):
            imp.zero_grad()
        self.model.zero_grad()
        torch.cuda.empty_cache()
        
        LOGGER.info('pruning done...')
        LOGGER.info(f'model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) Speed Up:{ori_flops / prune_flops:.2f}')
        LOGGER.info(f'model params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%)')
        LOGGER.info(f'model map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f})')
        LOGGER.info(f'model map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f})')
    
    def sparsity_learning(self, ckpt, world_size, prune):
        # Optimizer
        
        self.batch_size = self.batch_size // 2
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                            name=self.args.optimizer,
                                            lr=self.args.lr0,
                                            momentum=self.args.momentum,
                                            decay=weight_decay,
                                            iterations=iterations)
        
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.best_sl = {}
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')
        
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        # nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        nw = 100
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting Sparsity training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # Forward
                # with torch.cuda.amp.autocast(self.amp):
                batch = self.preprocess_batch(batch)
                self.loss, self.loss_items = self.model(batch)
                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                    else self.loss_items

                # Backward
                # self.scaler.scale(self.loss).backward()
                self.loss.backward()
                
                if isinstance(prune, (tp.pruner.BNScalePruner,)):
                    if self.args.reg_decay_type == 'linear':
                        reg = linear_trans(epoch, self.epochs, self.args.reg, self.args.reg * self.args.reg_decay)
                    elif self.args.reg_decay_type == 'step':
                        reg = self.args.reg * (self.args.reg_decay ** (epoch // self.args.reg_decay_step))
                    elif self.args.opt.reg_decay_type == 'constant':
                        reg = self.args.reg
                    prune.regularize(self.model, reg=reg)
                elif isinstance(prune, (tp.pruner.GroupNormPruner, tp.pruner.GrowingRegPruner)):
                    reg = self.args.reg
                    prune.regularize(self.model)

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            if isinstance(prune, (tp.pruner.GrowingRegPruner,)):
                prune.update_reg()
            
            if self.ema:
                model_sl = self.ema.ema.state_dict()
            else:
                model_sl = self.model.state_dict()
            
            tb = SummaryWriter(self.save_dir)
            
            bn_weight = []
            for name in model_sl:
                if 'weight' in name and len(model_sl[name].size()) == 1:
                    weight = model_sl[name].data.cpu().abs().clone().numpy().reshape((-1))
                    bn_weight.append(weight)
            bn_weight = np.concatenate(bn_weight)
            bn_weight = np.sort(bn_weight)
            bn_weight_percent = np.percentile(bn_weight, [1, 5, 10, 25, 50, 75])
            self.sparsity_ratio = np.sum(bn_weight < 1e-6) / bn_weight.shape[0]
            if f'{self.sparsity_ratio:.3f}' not in self.best_sl:
                self.best_sl[f'{self.sparsity_ratio:.3f}'] = 0.0
            if tb:
                tb.add_histogram('hist', bn_weight, epoch, bins='doane')
            
            del model_sl
            
            plt.figure(figsize=(15, 5), clear=True)
            plt.plot(bn_weight)
            plt.title(f'sparsity_ratio:{self.sparsity_ratio:.3f}\n')
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/visual/{epoch}_sl_{self.sparsity_ratio:.3f}.png')
            plt.close('all')
            LOGGER.info(f'epoch:{epoch} reg:{reg:.5f} sparsity_ratio:{self.sparsity_ratio:.5f} bn_weight_1:{bn_weight_percent[0]:.10f} bn_weight_5:{bn_weight_percent[1]:.8f} bn_weight_10:{bn_weight_percent[2]:.8f}\nbn_weight_25:{bn_weight_percent[3]:.5f} bn_weight_50:{bn_weight_percent[4]:.5f} bn_weight_75:{bn_weight_percent[5]:.5f}')
            
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                if self.fitness > self.best_sl[f'{self.sparsity_ratio:.3f}']:
                    self.best_sl[f'{self.sparsity_ratio:.3f}'] = self.fitness
                
                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
        
        sl = sorted(self.best_sl.keys(), key=lambda x:float(x))[-1]
        best_sl_model = self.wdir / 'best_sl_{}.pt'.format(sl)
        self.best = best_sl_model
        
        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
        return best_sl_model
    
    def compress(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(','))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0
        
        # init model
        ckpt = self.setup_model()
        replace_CSPLayer_with_CSPLayer_v2(self.model)
        # print(self.model)
        self.model = self.model.to(self.device)
        torch.save({'model':deepcopy(self.model), 'ema':None}, self.wdir / 'model_CSPLayer_v2.pt')
        self.set_model_attributes()
        
        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True
        
        # for sparsity learning
        self.amp = False
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()
        
        # get prune
        if type(self.args.imgsz) is int:
            example_inputs = torch.randn((1, 3, self.args.imgsz, self.args.imgsz)).to(self.device)
        elif type(self.args.imgsz) is list:
            example_inputs = torch.randn((1, 3, self.args.imgsz[0], self.args.imgsz[1])).to(self.device)
        else:
            assert f'self.args.imgsz type error! {self.args.imgsz}'
            
        # pre initt 
        batch_data = self.preprocess_batch(next(iter(self.train_loader), None))
        _, self.loss_items = self.model(batch_data)
        self.stopper = EarlyStopping(0)
        
        sparsity_learning, imp, prune = get_pruner(self.args, self.model, example_inputs)
        
        if sparsity_learning and not self.args.sl_model:
            self.args.sl_model = self.sparsity_learning(ckpt, world_size, prune)
        
        if sparsity_learning:
            model = torch.load(self.args.sl_model, map_location=self.device)
            model = model['ema' if model.get('ema') else 'model'].float()
            for p in model.parameters():
                p.requires_grad_(True)
            self.model = model
            self.model = self.model.to(self.device)
            self.set_model_attributes()
            sparsity_learning, imp, prune = get_pruner(self.args, self.model, example_inputs)
        
        # using for val
        if not hasattr(self, 'epoch'):
            self.epoch, self.epochs = 0, 1
        
        self.ema.ema = None
        self.model_prune(imp, prune, example_inputs)
        
        # test fuse
        fuse_model = deepcopy(self.model)
        for p in fuse_model.parameters():
            p.requires_grad_(False)
        print('fuse test...')
        fuse_model.fuse()
        print('fuse test done...')
        del fuse_model
        
        self.remove_all_hooks(self.model)
        prune_path = self.wdir / 'prune.pt'
        ckpt = {'model': deepcopy(de_parallel(self.model)), 'ema': None}
        torch.save(ckpt, prune_path, pickle_module=pickle)
        LOGGER.info(colorstr(f'Pruning after Finetune before the model is saved in:{prune_path}'))
        return str(prune_path)
    
    def __del__(self):
        self.train_loader = None
        self.test_loader = None
        self.model = None

class RTDETRFinetune(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = torch.load(self.args.model, map_location=self.device)
        model = model['ema' if model.get('ema') else 'model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        LOGGER.info(colorstr("prune_model info:"))
        model.info()
        return model
    
    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        # if str(model).endswith('.pt'):
        #     weights, ckpt = attempt_load_one_weight(model)
        #     cfg = ckpt['model'].yaml
        # else:
        #     cfg = model
        self.model = self.get_model(cfg=model, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return None