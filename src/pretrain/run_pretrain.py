import torch
import time
import random
import pprint
import math
from transformers import BertConfig, BertTokenizerFast
from src.modeling.modeling import ClipBertForPreTraining
from src.modeling.e2e_model import ClipBert

from src.datasets.dataset_pretrain import ClipBertPretrainDataset, PretrainCollator
from src.datasets.dataloader import MetaLoader, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import load_jsonl, load_json
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from collections import defaultdict
from tqdm import tqdm
from os.path import join
from apex import amp
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from src.utils.distributed import all_gather_list


def load_datalist_with_ratio(anno_path, data_ratio=1.0):
    raw_datalist = load_jsonl(anno_path)
    if data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * data_ratio)]
    return raw_datalist


def mk_vis_txt_pair_datalist(anno_path, data_ratio=1.0,
                             vis_id_key="coco_id", txt_key="caption"):
    """
    Args:
        anno_path: str, path to .jsonl file, each line is a dict
        data_ratio: float, (0, 1], when < 1, use part of the data.
        vis_id_key: str, image/video file id access key in the input dict.
        txt_key: str, txt access key in the input dict.

    Returns:

    """
    raw_datalist = load_datalist_with_ratio(anno_path, data_ratio)
    datalist = []
    for raw_d in raw_datalist:
        d = dict(
            txt=raw_d[txt_key],
            vis_id=raw_d[vis_id_key]
        )
        datalist.append(d)

    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vis_id"]].append(d)
    return grouped


def mk_captions_pretrain_dataloader(
        dataset_name, vis_format, anno_path, img_lmdb_dir, cfg, tokenizer, is_train=True):
    # make a list(dict), where each dict {vis_id: int, txt: str}
    if dataset_name == "coco_cap":
        grouped = mk_vis_txt_pair_datalist(
            anno_path, data_ratio=cfg.data_ratio,
            vis_id_key="coco_id", txt_key="caption")
    elif dataset_name == "vg_cap":
        grouped = mk_vis_txt_pair_datalist(
            anno_path, data_ratio=cfg.data_ratio,
            vis_id_key="vg_id", txt_key="caption")
    else:
        raise ValueError("Invalid dataset_name")

    # each group has a single image with multiple questions
    max_n_example_per_group = cfg.max_n_example_per_group \
        if vis_format == "image" else 1  # single element group for video.
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=max_n_example_per_group,
        is_train=is_train
    )

    frm_sampling_strategy = cfg.frm_sampling_strategy
    if not is_train and frm_sampling_strategy == "rand":
        frm_sampling_strategy = "middle"
    dataset = ClipBertPretrainDataset(
        datalist=group_datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=img_lmdb_dir,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        itm_neg_prob=cfg.itm_neg_prob,
        use_itm=cfg.use_itm,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        vis_format=vis_format
    )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, "
                f"group size {max_n_example_per_group}")
    batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    # hardcode video batch size to be 1 / num_frm of the image batch size.
    # so that video input image size could be similar to image batch size.
    batch_size = batch_size if vis_format == "image" else int(batch_size / cfg.num_frm)
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    data_collator = PretrainCollator(tokenizer=tokenizer,
                                     mlm=cfg.use_mlm,
                                     mlm_probability=0.15,
                                     max_length=cfg.max_txt_len,
                                     is_train=is_train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=data_collator.collate_batch)
    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loaders = {}
    for db in cfg.train_datasets:
        train_loaders[db.name] = mk_captions_pretrain_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, img_lmdb_dir=db.img,
            cfg=cfg, tokenizer=tokenizer, is_train=True
        )
        if "ratio" in db:
            train_loaders[db.name] = (train_loaders[db.name], db.ratio)

    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = mk_captions_pretrain_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, img_lmdb_dir=db.img,
            cfg=cfg, tokenizer=tokenizer, is_train=False
        )
    return train_loaders, val_loaders


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add pixel random sampling, only for pre-training
    model_cfg.pixel_random_sampling_size = cfg.pixel_random_sampling_size
    # add model-specific config
    add_attr_list = [
        "pixel_random_sampling_size",
    ]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])
    LOGGER.info(f"model_cfg {pprint.pformat(model_cfg.to_dict())}")

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = ClipBert(
        model_cfg, input_format=cfg.img_input_format,
        detectron2_model_cfg=cfg.detectron2_model_cfg,
        transformer_cls=ClipBertForPreTraining)
    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    else:
        LOGGER.info(f"Loading cnn weights from {cfg.detectron2_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            cnn_weights_path=cfg.detectron2_weights_path,
            bert_weights_path=cfg.bert_weights_path
        )

    if cfg.freeze_cnn:
        model.freeze_cnn_backbone()
    model.to(device)

    LOGGER.info("Setup model done!")
    return model


def forward_step(cfg, model, batch):
    """shared for training and validation"""
    # used to make visual feature copies
    if not cfg.use_itm:
        batch["itm_labels"] = None
    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()

    mlm_loss = 0
    n_mlm_tokens = 0
    n_mlm_corrects = 0
    itm_loss = 0
    n_itm_ex = 0
    n_itm_corrects = 0
    st = time.time()
    val_log = {'valid/mlm_loss': 0, 'valid/mlm_acc': 0,
               'valid/itm_loss': 0, 'valid/itm_acc': 0}
    debug_step = 5
    val_loaders = val_loader if isinstance(val_loader, dict) else {
        "unnamed_val_loader": val_loader}
    LOGGER.info(f"In total {len(val_loaders)} val loaders")
    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")
        for val_step, batch in enumerate(val_loader):
            # use iter to reset MetaLoader
            # forward pass
            outputs = forward_step(cfg, model, batch)

            # mlm
            mlm_labels = outputs["mlm_labels"]
            if cfg.use_mlm:
                mlm_loss += outputs["mlm_loss"].sum().item()
                mlm_mask = mlm_labels != -100  # (B, Lt)  -100 is the ignored label for cross entropy
                n_mlm_tokens += mlm_mask.sum().item()
                n_mlm_corrects += (
                        outputs["mlm_scores"][mlm_mask].max(
                            dim=-1)[1] == mlm_labels[mlm_mask]).sum().item()

            # itm
            if cfg.use_itm:
                itm_loss += outputs["itm_loss"].sum().item()
                n_itm_ex += len(outputs["itm_labels"])
                n_itm_corrects += (
                        outputs["itm_scores"].max(
                            dim=-1)[1] == outputs["itm_labels"]).sum().item()

            if cfg.debug and val_step >= debug_step:
                break
    # Gather across all processes
    mlm_loss = sum(all_gather_list(mlm_loss))
    n_mlm_corrects = sum(all_gather_list(n_mlm_corrects))
    n_mlm_tokens = sum(all_gather_list(n_mlm_tokens))
    itm_loss = sum(all_gather_list(itm_loss))
    n_itm_corrects = sum(all_gather_list(n_itm_corrects))
    n_itm_ex = sum(all_gather_list(n_itm_ex))

    if n_mlm_tokens != 0:
        val_log.update({
            'valid/mlm_loss': float(mlm_loss / n_mlm_tokens),
            'valid/mlm_acc': float(n_mlm_corrects / n_mlm_tokens)
        })
    if n_itm_ex != 0:
        val_log.update({
            'valid/itm_loss': float(itm_loss / n_itm_ex),
            'valid/itm_acc': float(n_itm_corrects / n_itm_ex)
        })

    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                f"[mlm_acc (per token)]: {val_log['valid/mlm_acc'] * 100:.2f} "
                f"[itm_acc (per example)]: {val_log['valid/itm_acc'] * 100:.2f} ")
    model.train()
    return val_log


def start_training():
    cfg = shared_configs.get_pretraining_args()
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info(f"device: {device} n_gpu: {n_gpu}, "
                f"rank: {hvd.rank()}, 16-bits training: {cfg.fp16}")

    model = setup_model(cfg, device=device)
    model.train()

    optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level='O2',
        keep_batchnorm_fp32=True)

    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    train_loaders, val_loaders = setup_dataloaders(cfg, tokenizer)
    train_loader = MetaLoader(train_loaders,
                              accum_steps=cfg.gradient_accumulation_steps,
                              distributed=n_gpu > 1)
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in val_loaders.items()}

    # compute the number of steps and update cfg
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    total_n_epochs = cfg.num_train_epochs
    cfg.num_train_steps = int(math.ceil(
        1. * train_loader.n_batches_in_epoch * total_n_epochs /
        (n_gpu * cfg.gradient_accumulation_steps)))
    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
        save_training_meta(cfg)
        path = join(
            cfg.output_dir, 'log', "detectron2_model_cfg.yaml")
        with open(path, "w") as f:
            f.write(model.cnn.config_file)
        LOGGER.info("Saving training done...")
        TB_LOGGER.create(join(cfg.output_dir, 'log'))
        pbar = tqdm(total=cfg.num_train_steps)
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
    LOGGER.info(f"  Total #batches - single epoch = {train_loader.n_batches_in_epoch}.")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Total #epochs = {total_n_epochs}.")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 5

    tasks = []
    for name, flag in zip(["mlm", "itm"], [cfg.use_mlm, cfg.use_itm]):
        if flag:
            tasks.append(name)
    task2loss = {t: RunningMeter(f'train_loss/{t}')
                 for t in tasks}
    task2loss["loss"] = RunningMeter('train_loss/loss')
    for step, (task, batch) in enumerate(train_loader):
        # forward pass
        outputs = forward_step(cfg, model, batch)
        mlm_loss, itm_loss = 0, 0
        if cfg.use_mlm:
            mlm_loss = outputs["mlm_loss"].mean()
            task2loss["mlm"](mlm_loss.item())
        if cfg.use_itm:
            itm_loss = outputs["itm_loss"].mean()
            task2loss["itm"](itm_loss.item())

        loss = mlm_loss + itm_loss
        task2loss["loss"](loss.item())

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            TB_LOGGER.log_scalar_dict({l.name: l.val
                                       for l in task2loss.values()
                                       if l.val is not None})
            n_epoch = int(1. * n_gpu * cfg.gradient_accumulation_steps *
                          global_step / train_loader.n_batches_in_epoch)
            # learning rate scheduling transformer
            lr_this_step_transformer = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)

            # learning rate scheduling cnn
            lr_this_step_cnn = get_lr_sched(
                global_step, cfg.cnn_lr_decay, cfg.cnn_learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.cnn_step_decay_epochs,
                multi_step_epoch=n_epoch)

            # Hardcoded param group length
            assert len(optimizer.param_groups) == 8
            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                if pg_n in [0, 1]:
                    param_group['lr'] = (
                        cfg.transformer_lr_mul * lr_this_step_transformer)
                elif pg_n in [2, 3]:
                    param_group['lr'] = lr_this_step_transformer
                elif pg_n in [4, 5]:
                    param_group['lr'] = (
                        cfg.cnn_lr_mul * lr_this_step_cnn)
                else:
                    param_group['lr'] = lr_this_step_cnn
            TB_LOGGER.add_scalar(
                "train/lr_transformer", lr_this_step_transformer,
                global_step)
            TB_LOGGER.add_scalar(
                "train/lr_cnn", lr_this_step_cnn, global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer), cfg.grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
            restorer.step()
            pbar.update(1)

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_loaders, cfg)
                model_saver.save(step=global_step, model=model)
        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_loaders, cfg)
        model_saver.save(step=global_step, model=model)


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    start_training()
