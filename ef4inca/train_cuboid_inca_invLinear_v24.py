#### Import modules!
import warnings
from typing import Union, Dict
from shutil import copyfile
from copy import deepcopy
import inspect
import pickle
import numpy as np
import glob
import random
import subprocess
import h5py
import os
import argparse
import logging

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from omegaconf import OmegaConf
# from einops import rearrange

from earthformer.config import cfg
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.metrics.sevir import SEVIRSkillScore #### These scores were not used in this study but kept here to guide future developments in this direction
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

from utils.visUtils_elegant import save_example_vis_results
from utils.fss_kucuk import compute_fss

from utils.fixedValues import NORM_DICT_TOTAL as normDict
from utils.fixedValues import bestSamples, worstSamples, randSamples

from utils.dataUtils_flex import kucukINCAdataModule



#### Set some directories/variables
_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(os.path.dirname(_curr_dir), "experiments") # Move one dir up and create a new directory, independent!

##### Any accelerator? ####
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')
                                   
pytorch_state_dict_name = "ef_inca_multisource2precip.pt"

class GradientNormLogger(Callback):
    def on_after_backward(self, trainer, pl_module):
        """
        Hook called after the backward pass to log the gradient norms per layer.
        """
        grad_norms_pre_clip = {}  # Store pre-clipping norms
        #grad_norms_post_clip = {}  # Store post-clipping norms
        total_norm_pre_clip = 0
        #total_norm_post_clip = 0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:  # Ensure the parameter has a gradient
                # Pre-clipping norm
                norm_pre_clip = param.grad.data.norm(2).item()
                grad_norms_pre_clip[f"grad_norm_pre_clip_{name.replace('.', '_')}"] = norm_pre_clip
                total_norm_pre_clip += norm_pre_clip ** 2

        # Compute total norms
        total_norm_pre_clip = total_norm_pre_clip ** 0.5

        # Log per-layer gradient norms
        for name, norm in grad_norms_pre_clip.items():
            pl_module.log(name, norm, on_step=True, on_epoch=False)

        # Log total gradient norms
        pl_module.log("grad_norm_total_pre_clip", total_norm_pre_clip, on_step=True, on_epoch=False)

        # Log the GradScaler's state (the scaling factor)
        if trainer.scaler is not None:
            scale_factor = trainer.scaler.get_scale()  # Get the current scale factor from the scaler
            pl_module.log("grad_scaler_scale", scale_factor, on_step=True, on_epoch=False)

class FinalGradientLogger(pl.Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total_norm = 0.0
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                total_norm += norm ** 2
                pl_module.log(f"grad_norm_final_{name.replace('.', '_')}", norm, on_step=True, on_epoch=False)
        total_norm = total_norm ** 0.5
        pl_module.log("grad_norm_total_final", total_norm, on_step=True, on_epoch=False)


class CuboidEF4INCAPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidEF4INCAPLModule, self).__init__()

        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only
        self.configure_save(cfg_file_path=oc_file)
        # evaluation
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.threshold_list = np.log10(np.array(self.threshold_list)) # thr's are given raw in the config file, but need to convert to log10!
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,)

        ## ADDITION FOR INVERSE LINEAR WEIGHTING OF LOSS FUNCTION! 
        self.wgt = self.invLinWeight(self.out_len)
        self.runFSS = oc.logging.computeFSS
        self.scale_list = oc.dataset.scale_list

         # Initialize a logger for file paths
        self.data_logger = self._initialize_data_logger()
        self.NaN_logger = self._initialize_NaN_logger()

    def invLinWeight(self, t):
        '''Rationale is to inversely scale the loss values across time dimension, i.e., with [24, 23, 22, ..., 1] for a 24-step prediction.
        This forces the model to pay extra attention to earlier lead times.
        '''
        self.wgt = []
        for i in range(t):
            self.wgt.append(t-i)
        self.wgt = np.array(self.wgt)
        return torch.from_numpy(self.wgt.astype(np.float32)).to(device).reshape((-1, 1, 1, 1))

    def _initialize_data_logger(self):
        """Initialize a separate logger for logging data being processed."""
        log_file_path = os.path.join(self.save_dir, "data_processing.log")
        logger = logging.getLogger("DataLogger")
        logger.setLevel(logging.INFO)
        if not logger.handlers:  # Avoid duplicate handlers
            handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _initialize_NaN_logger(self):
        """Initialize a separate logger for logging NaN data being processed."""
        log_file_path = os.path.join(self.save_dir, "data_NaN.log")
        logger = logging.getLogger("NaNLogger")
        logger.setLevel(logging.INFO)
        if not logger.handlers:  # Avoid duplicate handlers
            handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        self.bulkScores_dir = os.path.join(self.save_dir, 'bulkScores')
        os.makedirs(self.bulkScores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)
        self.testOutput_save_dir = os.path.join(self.save_dir, "testOutput") ## Directory for saving predictions of test samples!
        os.makedirs(self.testOutput_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.dataset = self.get_dataset_config()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_dataset_config():
        oc = OmegaConf.create()
        oc.dataset_name = "inca"
        oc.img_height = 400
        oc.img_width = 700
        oc.in_len = 25
        oc.out_len = 24
        oc.seq_len = 49
        oc.plot_stride = 2
        oc.interval_real_time = 5
        oc.sample_mode = "sequent"
        oc.stride = oc.out_len
        oc.layout = "NTHWC"
        oc.start_date = None
        oc.end_date = None
        oc.metrics_mode = "0"
        oc.metrics_list = ('csi', 'pod', 'sucr', 'bias')
        oc.threshold_list = (0.6, 1.7, 2.7, 5, 8.6, 15)
        return oc
    
    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'
        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
        cfg.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_layout_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.in_len = dataset_oc.in_len
        oc.out_len = dataset_oc.out_len
        oc.layout = dataset_oc.layout
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 100
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Has to be 'cosine' in the current implementation
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 20
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "EF4INCA"
        oc.monitor_lr = True
        oc.monitor_device = False
        # oc.track_grad_norm = -1
        oc.use_wandb = True
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = "16-mixed"
        return oc

    @classmethod
    def get_vis_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [80, ]
        oc.test_example_data_idx_list = [0, 80, 160, 240, 320, 400]
        oc.eval_example_only = False
        oc.plot_stride = dataset_oc.plot_stride
        return oc


    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n not in decay_parameters],
            'weight_decay': 0.0
        }]
    
        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError
    
        # Modified OneCycleLR
        lr_scheduler = OneCycleLR(optimizer,
                                  max_lr=self.oc.optim.lr,
                                  total_steps=20209, #
                                  div_factor = 25,
                                  pct_start=self.oc.optim.warmup_percentage,
                                  final_div_factor= 10,  
                                  three_phase = False,
                                  anneal_strategy='cos'  
                                 )

        lr_scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
        }
    
        return [optimizer], [lr_scheduler_config]


    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        # Add Gradient Norm Logger callback
        callbacks += [GradientNormLogger()]
        callbacks += [FinalGradientLogger()]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ##
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="auto",
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret
    

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_inca_Datamodule(dataset_oc,
                             ):
        ####
        dm = kucukINCAdataModule(
            params=dataset_oc,
            data_dir=os.path.dirname(_curr_dir)+'/data/'
        )
        return dm

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    def forward(self, in_seq, out_seq):
        """
        This is modified to inversely scale the loss values across time dimension!
        Loss function is kept the same as the original!
        """
        output = self.torch_nn_module(in_seq)
        ## Keep loss values per pixel and time step!
        loss = F.mse_loss(output, out_seq, reduction="none")
        loss = torch.mean(loss)
        return output, loss

    # def forward(self, in_seq, out_seq):
    # """
    # Compute the Balanced Loss with inverse time weighting applied before calculating B-MSE and B-MAE.
    # """
    # # Get model predictions
    # output = self.torch_nn_module(in_seq)

    # # Compute the weights based on observed rainfall intensity
    # R = out_seq  # Observed rainfall intensity
    # spatial_weights = torch.where(
    #     R <= 1,
    #     torch.tensor(1.0, device=R.device),
    #     torch.where(
    #         R <= 30,
    #         R,
    #         torch.tensor(30.0, device=R.device)
    #     )
    # )

    # # Apply inverse time weighting
    # inverse_weights = self.wgt.to(self.device)  # Shape: [T, 1, 1, 1]

    # # Compute B-MSE with time weighting
    # mse_loss = torch.mean(inverse_weights * spatial_weights * (output - out_seq) ** 2)

    # # Compute B-MAE with time weighting
    # mae_loss = torch.mean(inverse_weights * spatial_weights * torch.abs(output - out_seq))

    # # Balanced Loss (B-MSE + B-MAE)
    # loss = mse_loss + mae_loss

    # return output, loss


    # def forward(self, in_seq, out_seq):
    #     """
    #     Compute the Balanced Loss (B-MSE + B-MAE) for the model predictions.
    #     Args:
    #         in_seq: Input sequence to the model (e.g., past rainfall data).
    #         out_seq: Ground truth (observed rainfall intensity).
    #     Returns:
    #         output: Model predictions.
    #         loss: Balanced loss value.
    #     """
    #     # Get model predictions
    #     output = self.torch_nn_module(in_seq)
        
    #     # Compute the weights based on observed rainfall intensity
    #     # R = 10 ** out_seq  # Observed rainfall intensity transformed back to real values
    #     R = torch.pow(10, out_seq) # Observed rainfall intensity transformed back to real values
    #     max_weights = 10
    #     weights = torch.where(
    #         R <= 1, 
    #         torch.tensor(1.0, device=R.device),
    #         torch.where(
    #             R <= max_weights, 
    #             R, 
    #             torch.tensor(max_weights, device=R.device)
    #         )
    #     )
        
    #     # Compute B-MSE
    #     mse_loss = torch.mean(weights * (output - out_seq) ** 2)
        
    #     # Compute B-MAE
    #     mae_loss = torch.mean(weights * torch.abs(output - out_seq))
        
    #     # Combined Balanced Loss
    #     loss = mse_loss + mae_loss
        
    #     return output, loss


    def training_step(self, batch, batch_idx):
        x = batch['sample_past'].contiguous()
        y = batch['sample_future'].contiguous()
        tmp_name = batch['name'] 
        ##
        y_hat, loss = self(x, y)
        micro_batch_size = x.shape[self.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)
        
        self.data_logger.info(f"Processing data_idx: {data_idx}, file(s): {tmp_name}, Loss: {loss.item()}")

        if torch.isnan(x).any():
            print(f'NaN detected in input sequence with file(s): {tmp_name}')
            
        if torch.isinf(x).any():
            print(f'Inf detected in input sequence with file(s): {tmp_name}')

        if torch.isnan(y).any():
            print(f'NaN detected in target sequence with file(s): {tmp_name}')

        if torch.isinf(y).any():
            print(f'Inf detected in target sequence with file(s): {tmp_name}')
        
        self.save_vis_step_end(
            data_idx=data_idx,
            in_seq=x,
            target_seq=y,
            pred_seq=y_hat,
            mode="train",
            title=tmp_name
        )
        self.log('train_loss', loss,
                 on_step=True, on_epoch=False)
        self.log('learning_rate', self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], #### Radical!
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x = batch['sample_past'].contiguous()
        y = batch['sample_future'].contiguous()
        tmp_name = batch['name']  # Assume this contains the file names or identifiers
        ##
        micro_batch_size = x.shape[self.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)

        target_nan = []
        pred_nan = []

        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            y_hat, _ = self(x, y)
            self.save_vis_step_end(
                data_idx=data_idx,
                in_seq=x,
                target_seq=y,
                pred_seq=y_hat,
                mode="val",
                title=tmp_name
            )

                    # Check for NaN or Inf in predictions/targets
            if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
                print(f"NaN or Inf detected in predictions at data_idx {data_idx}, file(s): {tmp_name}")
                #pred_nan.append(data_idx)
                self.save_vis_step_end(
                    data_idx=data_idx,
                    in_seq=x,
                    target_seq=y,
                    pred_seq=y_hat,
                    mode="val",
                    title=tmp_name
                )                 
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"NaN or Inf detected in targets at data {data_idx}, file(s): {tmp_name}")
                #target_nan.append(data_idx)
                self.save_vis_step_end(
                    data_idx=data_idx,
                    in_seq=x,
                    target_seq=y,
                    pred_seq=y_hat,
                    mode="val",
                    title=tmp_name
                )    

            
            # Step-level metrics (functional call)
            step_mse = self.valid_mse(y_hat, y)  # Batch MSE
            step_mae = self.valid_mae(y_hat, y)  # Batch MAE
    
            # Epoch-level metrics (stateful updates)
            self.valid_mse.update(y_hat, y)  # Update cumulative MSE
            self.valid_mae.update(y_hat, y)  # Update cumulative MAE
            self.valid_score.update(y_hat, y)
    
            # Log step-level metrics
            self.log('valid_frame_mse_step', step_mse,
                     prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_frame_mae_step', step_mae,
                     prog_bar=True, on_step=True, on_epoch=False)
        return None

    def on_validation_epoch_end(self):
        print(f'Current learning rate is: {self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]}')
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log('valid_frame_mse_epoch', valid_mse,
                 prog_bar=True, on_step=False, on_epoch=True)  # , sync_dist=True
        self.log('valid_frame_mae_epoch', valid_mae,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        valid_score = self.valid_score.compute()
        self.log("valid_loss_epoch", -valid_score["avg"]["csi"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log_score_epoch_end(score_dict=valid_score, mode="val")
        self.valid_score.reset()
        self.save_score_epoch_end(score_dict=valid_score,
                                  mse=valid_mse,
                                  mae=valid_mae,
                                  mode="val")


    def test_step(self, batch, batch_idx):
        x = batch['sample_past'].contiguous()
        y = batch['sample_future'].contiguous()
        tmp_name = batch['name']

        micro_batch_size = x.shape[self.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)

        target_nan = []
        pred_nan = []
        
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            y_hat, _ = self(x, y)
            if self.runFSS:
                compute_fss(y=y, y_hat=y_hat, tmp_name=tmp_name, save_dir=self.bulkScores_dir,
                            threshold_list=self.threshold_list,
                            scale_list=self.scale_list, 
                            device=device,)
            self.save_vis_step_end(
                data_idx=data_idx,
                in_seq=x,
                target_seq=y,
                pred_seq=y_hat,
                mode="test",
                title=tmp_name
            )

                    # Check for NaN or Inf in predictions/targets
            if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
                print(f"NaN or Inf detected in predictions at data_idx {data_idx}, file(s): {tmp_name}")
                #pred_nan.append(data_idx)
                self.save_vis_step_end(
                    data_idx=data_idx,
                    in_seq=x,
                    target_seq=y,
                    pred_seq=y_hat,
                    mode="val",
                    title=tmp_name
                )                 
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"NaN or Inf detected in targets at data {data_idx}, file(s): {tmp_name}")
                #target_nan.append(data_idx)
                self.save_vis_step_end(
                    data_idx=data_idx,
                    in_seq=x,
                    target_seq=y,
                    pred_seq=y_hat,
                    mode="val",
                    title=tmp_name
                )  
                
            step_mse = self.test_mse(y_hat, y)
            step_mae = self.test_mae(y_hat, y)
            self.test_score.update(y_hat, y)
            self.log('test_frame_mse_step', step_mse,
                     prog_bar=True, on_step=True, on_epoch=False)
            self.log('test_frame_mae_step', step_mae,
                     prog_bar=True, on_step=True, on_epoch=False)
            ## Save predictions for selected samples!
            shortName = tmp_name[0].split('/')[-1][:-3] 
            # if shortName in bestSamples or shortName in worstSamples or shortName in randSamples:
                    #Save it!
            assert y.shape[0] == 1, "Not implemented for batch size>1!"
            ## Ensure the tensors are in float32
            y = y.to(torch.float32)
            y_hat = y_hat.to(torch.float32)
            # Denorm and detach while backtransforming!
            y = torch.pow(10,y).detach().cpu().numpy()
            y_hat = torch.pow(10,y_hat).detach().cpu().numpy()
            with h5py.File(self.testOutput_save_dir + '/' + shortName + '.h5', 'w') as hf:
                hf.create_dataset('y', data=y[0,:,:,:,0])
                hf.create_dataset('y_hat', data=y_hat[0,:,:,:,0])
        return None

    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        self.log('test_frame_mse_epoch', test_mse,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_frame_mae_epoch', test_mae,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        test_score = self.test_score.compute()
        self.log_score_epoch_end(score_dict=test_score, mode="test")
        self.test_score.reset()
        self.save_score_epoch_end(score_dict=test_score,
                                  mse=test_mse,
                                  mae=test_mae,
                                  mode="test")

    def log_score_epoch_end(self, score_dict: Dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.metrics_list:
            for thresh in self.threshold_list:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_{thresh}_epoch", score_mean, sync_dist=True) # Careful if using multiple gpus...
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_avg_epoch", score_avg_mean, sync_dist=True)  

    def save_score_epoch_end(self,
                             score_dict: Dict,
                             mse: Union[np.ndarray, float],
                             mae: Union[np.ndarray, float],
                             mode: str = "val"):
        assert mode in ["val", "test"], f"Wrong mode {mode}. Must be 'val' or 'test'."
        if self.local_rank == 0:
            save_dict = deepcopy(score_dict)
            save_dict.update(dict(mse=mse, mae=mae))
            if self.scores_dir is not None:
                save_path = os.path.join(self.scores_dir, f"{mode}_results_epoch_{self.current_epoch}.pkl")
                f = open(save_path, 'wb')
                pickle.dump(save_dict, f)
                f.close()

    def save_vis_step_end(
            self,
            data_idx: int,
            in_seq: torch.Tensor,
            target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            mode: str = "train",
            title: str = None,):
        r"""
        Parameters
        ----------
        data_idx:   int
            data_idx == batch_idx * micro_batch_size
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if data_idx in example_data_idx_list:
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    title=title,
                    label=self.oc.logging.logging_prefix,
                    )

    def save_vis_step_NaN_Inf(
        self,
        data_idx: int,
        in_seq: torch.Tensor,
        target_seq: torch.Tensor,
        pred_seq: torch.Tensor,
        mode: str = "train",
        title: str = None,
    ):
        r"""
        Parameters
        ----------
        data_idx:   int
            data_idx == batch_idx * micro_batch_size
        """
        if self.local_rank == 0:
            if torch.isnan(in_seq).any() or torch.isinf(in_seq).any or torch.isnan(target_seq).any() or toch.isinf(target_seq).any():
                print(f"NaN or inf detected in in_seq or target_seq at data_idx {data_idx}. Saving visual results.")
                
                # Save the visual results
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    title=title,
                    label=self.oc.logging.logging_prefix,
                )
                # else:
                #     print(f"No NaN detected in in_seq at data_idx {data_idx}. Skipping save.")

def parse_gpu_list(value):
    if ',' in value:
        return [int(x) for x in value.split(',')]
    else:
        return [int(value)]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str,
        help='Provide a name here if you want to save the results in a subdirectory under the experiments directory will be created anyway.',
        default='repotest')  # Prefix of the save dir
    parser.add_argument('--cfg', default='cfgs/cfg4inca.yaml', type=str)
    # parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--gpus', type=parse_gpu_list, default=[0], help="GPU indices to use, e.g., '1' or '0,1'") # check if this works
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser.add_argument('--ckpt_name', 
        default='last.ckpt', 
        type=str, help='The model checkpoint trained for EF4INCA.')
    args = parser.parse_args()
    return parser

def main():
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_oc = OmegaConf.to_object(CuboidEF4INCAPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * len(args.gpus))
        max_epochs = None
        seed = 0
    seed_everything(seed, workers=True)
    dm = CuboidEF4INCAPLModule.get_inca_Datamodule(
        dataset_oc=dataset_oc,
        )
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * len(args.gpus)) #check if this is still working when using multiple GPU's
    total_num_steps = CuboidEF4INCAPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = CuboidEF4INCAPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)
    pl_module.to(device) # Let's see if this works...
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    theCommand = "cp -r ../ef4inca "+pl_module.save_dir
    os.system(theCommand) # It's good to have a copy of the code in the save directory!
    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        pretrained_ckpt_name = pytorch_state_dict_name
        print(os.path.join(os.path.dirname(_curr_dir), 'trained_ckpt', pretrained_ckpt_name))
        state_dict = torch.load(os.path.join(os.path.dirname(_curr_dir), 'trained_ckpt', pretrained_ckpt_name),
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm)
    elif args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
            print('Device used :', device)
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        state_dict = pl_ckpt_to_pytorch_state_dict(checkpoint_path=trainer.checkpoint_callback.best_model_path,
                                                   map_location=torch.device("cpu"),
                                                   delete_prefix_len=len("torch_nn_module."))
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        trainer.test(ckpt_path="best",
                     datamodule=dm)
        
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main()
