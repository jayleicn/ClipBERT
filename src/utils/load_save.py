"""
saving utilities
"""
import json
import os
from os.path import dirname, exists, join, realpath
import subprocess
from apex import amp
from easydict import EasyDict as edict

import torch
from src.utils.basic_utils import save_json, make_zipfile, load_json
from src.utils.logger import LOGGER
from typing import Any, Dict, Union


def save_training_meta(args):
    # args is an EasyDict object, treat it the same as a normal dict
    os.makedirs(join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(join(args.output_dir, 'ckpt'), exist_ok=True)

    # training args
    save_args_path = join(args.output_dir, 'log', 'args.json')
    save_json(args, save_args_path, save_pretty=True)

    # model args
    model_config = json.load(open(args.model_config))
    save_model_config_path = join(args.output_dir, 'log', 'model_config.json')
    save_json(model_config, save_model_config_path, save_pretty=True)

    # save a copy of the codebase. !!!Do not store heavy file in your codebase when using it.
    code_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
    code_zip_filename = os.path.join(args.output_dir, "code.zip")
    LOGGER.info(f"Saving code from {code_dir} to {code_zip_filename}...")
    make_zipfile(code_dir, code_zip_filename,
                 enclosing_dir="code",
                 exclude_dirs_substring="results",
                 exclude_dirs=["results", "debug_results", "__pycache__"],
                 exclude_extensions=[".pyc", ".ipynb", ".swap"])
    LOGGER.info(f"Saving code done.")


class ModelSaver(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.max_save_load_trial = 10

    def save(self, step, model, optimizer=None, prefix="model"):
        model_path = join(self.output_dir, f"{prefix}_step_{step}.pt")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        # with retrial, as azure blob fails occasionally.
        save_trial = 0
        while save_trial < self.max_save_load_trial:
            try:
                LOGGER.info(f"ModelSaver save trial NO. {save_trial}")
                torch.save(state_dict, model_path)
                if optimizer is not None:
                    optimizer_state_dict = \
                        {k: v.cpu() if isinstance(v, torch.Tensor) else v
                         for k, v in optimizer.state_dict().items()}
                    dump = {'step': step, 'optimizer': optimizer_state_dict}
                    torch.save(
                        dump,
                        f'{self.output_dir}/{prefix}_step_{step}_train_state.pt')
                break
            except Exception as e:
                save_trial += 1


def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
    """operated in-place, no need to return `model`"""

    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(
            loaded_state_dict_or_path, map_location="cpu")
    else:
        loaded_state_dict = loaded_state_dict_or_path
    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]

    LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
    LOGGER.info("Keys in loaded but not in model:")
    diff_keys = load_keys.difference(model_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model but not in loaded:")
    diff_keys = model_keys.difference(load_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model and loaded, but shape mismatched:")
    LOGGER.info(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
    model.load_state_dict(toload, strict=False)


def compare_dict_difference(dict1, dict2, dict1_name="dict1",
                            dict2_name="dict2",
                            print_value_diff=True, verbose=False):
    """
    Args:
        dict1:
        dict2:
        dict1_name:
        dict2_name:
        print_value_diff: bool, output dict value difference within shared keys
            for dict1 and dict2. In effect only when verbose == True
        verbose:
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    shared_keys = keys1.intersection(keys2)
    keys1_unique = keys1.difference(shared_keys)
    keys2_unique = keys2.difference(shared_keys)
    key_diff_list = list(keys1_unique) + list(keys2_unique)

    # value difference in the shared keys in dict1 and dict2
    value_diff_dict = {}
    for k in shared_keys:
        if dict1[k] != dict2[k]:
            value_diff_dict[k] = [(dict1_name, dict1[k]), (dict2_name, dict2[k])]

    if verbose:
        LOGGER.info("=" * 30 + "key difference")
        LOGGER.info(f"keys in {dict1_name} but not in {dict2_name}: "
                    f"total {len(keys1_unique)}, {sorted(keys1_unique)}")
        LOGGER.info(f"keys in {dict2_name} but not in {dict1_name}: "
                    f"total {len(keys2_unique)}, {sorted(keys2_unique)}")

    if verbose and print_value_diff:

        LOGGER.info("=" * 30 + "value difference")
        LOGGER.info(f"{json.dumps(value_diff_dict, indent=4)}")

    return value_diff_dict, key_diff_list


def _to_cuda(state):
    """ usually load from cpu checkpoint but need to load to cuda """
    if isinstance(state, torch.Tensor):
        ret = state.cuda()  # assume propoerly set py torch.cuda.set_device
        if 'Half' in state.type():
            ret = ret.float()  # apex O2 requires it
        return ret
    elif isinstance(state, list):
        new_state = [_to_cuda(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cuda(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cuda(t) for n, t in state.items()}
    else:
        return state
    return new_state


def _to_cpu(state):
    """ store in cpu to avoid GPU0 device, fp16 to save space """
    if isinstance(state, torch.Tensor):
        ret = state.cpu()
        if 'Float' in state.type():
            ret = ret.half()
        return ret
    elif isinstance(state, list):
        new_state = [_to_cpu(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cpu(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cpu(t) for n, t in state.items()}
    else:
        return state
    return new_state


class TrainingRestorer(object):
    """ckpt_dict: a dict contains all optimizers/models"""
    def __init__(self, opts, **ckpt_dict):
        if exists(opts.output_dir):
            restore_opts = json.load(open(
                f'{opts.output_dir}/log/args.json', 'r'))
            assert opts == edict(restore_opts)
        # keep 2 checkpoints in case of corrupted
        self.save_path = f'{opts.output_dir}/restore.pt'
        self.backup_path = f'{opts.output_dir}/restore_backup.pt'
        self.ckpt_dict = ckpt_dict
        self.save_steps = opts.save_steps
        self.amp = opts.fp16
        # since saving to or loading from azure blob fails sometimes
        self.max_save_load_trial = 10
        if exists(self.save_path) or exists(self.backup_path):
            LOGGER.info('found previous checkpoint. try to resume...')
            # with retrial, as azure blob fails occasionally.
            restore_trial = 0
            while restore_trial < self.max_save_load_trial:
                LOGGER.info(f"TrainingRestorer restore trial NO. {restore_trial}")
                try:
                    self.restore()
                    break
                except Exception as e:
                    restore_trial += 1
        else:
            self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step % self.save_steps == 0:
            # with retrial, as azure blob fails occasionally.
            save_trial = 0
            while save_trial < self.max_save_load_trial:
                LOGGER.info(f"TrainingRestorer save trial NO. {save_trial}")
                try:
                    self.save()
                    break
                except Exception as e:
                    save_trial += 1

    def save(self):
        checkpoint_to_save = {'global_step': self.global_step}
        for k in self.ckpt_dict:
            checkpoint_to_save[k] = _to_cpu(self.ckpt_dict[k].state_dict())
        if self.amp:
            checkpoint_to_save['amp_state_dict'] = amp.state_dict()
        if exists(self.save_path):
            os.rename(self.save_path, self.backup_path)
        torch.save(checkpoint_to_save, self.save_path)

    def restore(self):
        try:
            checkpoint = torch.load(self.save_path)
        except Exception:
            checkpoint = torch.load(self.backup_path)
        self.global_step = checkpoint['global_step']
        for k in self.ckpt_dict:
            self.ckpt_dict[k].load_state_dict(_to_cuda(checkpoint[k]))
        if self.amp:
            amp.load_state_dict(checkpoint['amp_state_dict'])
        LOGGER.info(f'resume training from step {self.global_step}')


class E2E_TrainingRestorer(object):
    def __init__(self, opts, model, optimizer):
        if exists(f"{opts.output_dir}/log/args.json"):
            restore_opts = json.load(
                open(f'{opts.output_dir}/log/args.json', 'r'))
            with open(join(
                    opts.output_dir, 'log',
                    'restore_args.json'), 'w') as writer:
                json.dump(vars(opts), writer, indent=4)
            # assert opts == edict(restore_opts)
        # keep 2 checkpoints in case of corrupted
        self.save_path = f'{opts.output_dir}/restore.pt'
        self.backup_path = f'{opts.output_dir}/restore_backup.pt'
        self.model = model
        self.optimizer = optimizer
        self.save_steps = int(opts.save_steps_ratio * opts.num_train_steps)
        self.amp = opts.fp16
        # since saving to or loading from azure blob fails sometimes
        self.max_save_load_trial = 10
        if exists(self.save_path) or exists(self.backup_path):
            LOGGER.info('found previous checkpoint. try to resume...')
            # with retrial, as azure blob fails occasionally.
            restore_trial = 0
            while restore_trial < self.max_save_load_trial:
                LOGGER.info(f"TrainingRestorer restore trial NO. {restore_trial}")
                try:
                    self.restore(opts)
                    break
                except Exception as e:
                    restore_trial += 1
        else:
            self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step % self.save_steps == 0:
            # with retrial, as azure blob fails occasionally.
            save_trial = 0
            while save_trial < self.max_save_load_trial:
                LOGGER.info(f"TrainingRestorer save trial NO. {save_trial}")
                try:
                    self.save()
                    break
                except Exception as e:
                    save_trial += 1

    def save(self):
        checkpoint = {'global_step': self.global_step,
                      'model_state_dict': _to_cpu(self.model.state_dict()),
                      'optim_state_dict': _to_cpu(self.optimizer.state_dict())}
        if self.amp:
            checkpoint['amp_state_dict'] = amp.state_dict()
        if exists(self.save_path):
            os.rename(self.save_path, self.backup_path)
        torch.save(checkpoint, self.save_path)

    def restore(self, opts):
        try:
            checkpoint = torch.load(self.save_path)
        except Exception:
            checkpoint = torch.load(self.backup_path)
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(_to_cuda(checkpoint['model_state_dict']))
        self.optimizer.load_state_dict(
            _to_cuda(checkpoint['optim_state_dict']))
        if self.amp:
            amp.load_state_dict(checkpoint['amp_state_dict'])
        LOGGER.info(f'resume training from step {self.global_step}')


def convert_torchvision_ckpt_to_detectron2(ckpt_path) -> Dict[str, Any]:
    r"""
    Return state dict of visual backbone which can be loaded with
    `Detectron2 <https://github.com/facebookresearch/detectron2>`_.
    This is useful for downstream tasks based on Detectron2 (such as
    object detection and instance segmentation). This method renames
    certain parameters from Torchvision-style to Detectron2-style.
    Returns
    -------
    Dict[str, Any]
        A dict with three keys: ``{"model", "author", "matching_heuristics"}``.
        These are necessary keys for loading this state dict properly with
        Detectron2.
    """
    assert os.path.exists(ckpt_path)
    input_state_dict = torch.load(ckpt_path)

    # Detectron2 backbones have slightly different module names, this mapping
    # lists substrings of module names required to be renamed for loading a
    # torchvision model into Detectron2.
    DETECTRON2_RENAME_MAPPING: Dict[str, str] = {
        "layer1": "res2",
        "layer2": "res3",
        "layer3": "res4",
        "layer4": "res5",
        "bn1": "conv1.norm",
        "bn2": "conv2.norm",
        "bn3": "conv3.norm",
        "downsample.0": "shortcut",
        "downsample.1": "shortcut.norm",
    }
    # Populate this dict by renaming module names.
    d2_backbone_dict: Dict[str, torch.Tensor] = {}

    for name, param in input_state_dict.items():
        for old, new in DETECTRON2_RENAME_MAPPING.items():
            name = name.replace(old, new)

        # First conv and bn module parameters are prefixed with "stem.".
        if not name.startswith("res"):
            name = f"stem.{name}"

        d2_backbone_dict[name] = param

    return {
        "model": d2_backbone_dict,
        "__author__": "Karan Desai",
        "matching_heuristics": True,
    }
