import torch
import os
import time
import random
import math
from transformers import BertConfig, BertTokenizerFast
from src.modeling.modeling import ClipBertForVideoTextRetrieval

from src.modeling.e2e_model import ClipBert

from src.datasets.dataset_video_retrieval import (
    ClipBertVideoRetrievalDataset, VideoRetrievalCollator, ClipBertVideoRetrievalEvalDataset)
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json, get_rounded_percentage)
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer

import numpy as np
from tqdm import tqdm
from os.path import join, exists
from easydict import EasyDict as edict
from apex import amp
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from src.utils.distributed import all_gather_list
from collections import defaultdict


def mk_video_ret_datalist(raw_datalist, cfg):
    """
    Args:
        raw_datalist: list(dict)
        cfg:

    Returns:

    """
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    if cfg.data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]
        LOGGER.info(f"Use {100 * cfg.data_ratio}% of the loaded data: {len(raw_datalist)}")

    datalist = []
    qid = 0
    for raw_d in raw_datalist:
        d = dict(
            id=qid,
            txt=raw_d["caption"],
            vid_id=raw_d["clip_name"]
        )
        qid += 1
        datalist.append(d)
    LOGGER.info(f"datalist {len(datalist)}")
    return datalist


def mk_video_ret_dataloader(anno_path, lmdb_dir, cfg, tokenizer, is_train=True):
    """"""
    raw_datalist = load_jsonl(anno_path)
    datalist = mk_video_ret_datalist(raw_datalist, cfg)
    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vid_id"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")

    frm_sampling_strategy = cfg.frm_sampling_strategy
    if not is_train and frm_sampling_strategy == "rand":
        frm_sampling_strategy = "middle"
    dataset = ClipBertVideoRetrievalDataset(
        datalist=group_datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        itm_neg_size=cfg.itm_neg_size,
        ensemble_n_clips=cfg.train_n_clips,
        random_sample_clips=cfg.random_sample_clips
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    vqa_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vqa_collator.collate_batch)
    return dataloader


def mk_video_ret_eval_dataloader(anno_path, lmdb_dir, cfg, tokenizer):
    """
    eval_retrieval: bool, will sample one video per batch paired with multiple text.
    Returns:

    """
    raw_datalist = load_jsonl(anno_path)
    datalist = mk_video_ret_datalist(raw_datalist, cfg)
    frm_sampling_strategy = cfg.frm_sampling_strategy
    if frm_sampling_strategy == "rand":
        frm_sampling_strategy = "middle"
    dataset = ClipBertVideoRetrievalEvalDataset(
        datalist=datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        ensemble_n_clips=cfg.inference_n_clips,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=False)
    retrieval_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=1,  # already batched in dataset
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=retrieval_collator.collate_batch)
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    dataloader = PrefetchLoader(dataloader, img_norm)
    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loader = mk_video_ret_dataloader(
        anno_path=cfg.train_datasets[0].txt,
        lmdb_dir=cfg.train_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=True
    )
    val_loader = mk_video_ret_dataloader(
        anno_path=cfg.val_datasets[0].txt,
        lmdb_dir=cfg.val_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=False
    )
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loader = PrefetchLoader(val_loader, img_norm)
    return train_loader, val_loader


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    add_attr_list = [
        "num_labels", "classifier", "cls_hidden_scale",
        "loss_type", "margin",
    ]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])
    transformer_model_cls = ClipBertForVideoTextRetrieval

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = ClipBert(
        model_cfg, input_format=cfg.img_input_format,
        detectron2_model_cfg=cfg.detectron2_model_cfg,
        transformer_cls=transformer_model_cls)
    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    else:
        LOGGER.info(f"Loading cnn weights from {cfg.detectron2_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            cnn_weights_path=cfg.detectron2_weights_path,
            bert_weights_path=cfg.bert_weights_path)
    if cfg.freeze_cnn:
        model.freeze_cnn_backbone()
    model.to(device)

    LOGGER.info("Setup model done!")
    return model


def forward_step(model, batch, cfg):
    """shared for training and validation"""
    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def validate(model, val_loader, eval_loader, cfg, train_global_step, eval_filepath):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    model.eval()

    loss = 0.
    n_ex = 0
    n_corrects = 0
    st = time.time()
    debug_step = 5
    for val_step, batch in enumerate(val_loader):
        # forward pass
        del batch["caption_ids"]
        outputs = forward_step(model, batch, cfg)
        targets = batch['labels']

        loss += outputs["loss"].sum().item() if isinstance(
            outputs["loss"], torch.Tensor) else 0
        n_ex += len(targets)

        if outputs["logits"].shape[1] == 2:
            n_corrects += (
                outputs["logits"].max(
                    dim=-1)[1] == targets).sum().item()
        else:
            predictions = (
                torch.sigmoid(outputs["logits"]) > 0.5).long()
            predictions = predictions.view(outputs["loss"].shape[0], -1)
            targets = targets.view(outputs["loss"].shape[0], -1)
            matched = predictions[:, 0].squeeze() == targets[:, 0].squeeze()
            n_corrects += matched.sum().item()

        if cfg.debug and val_step >= debug_step:
            break

    loss = sum(all_gather_list(loss))
    n_ex = sum(all_gather_list(n_ex))
    n_corrects = sum(all_gather_list(n_corrects))

    _, retrieval_metrics = inference_retrieval(model, eval_loader, eval_filepath, cfg)

    model.train()

    if hvd.rank() == 0:
        # average loss for each example
        acc = float(n_corrects / n_ex)
        val_log = {'valid/loss': float(loss / n_ex), 'valid/acc': acc}
        for ret_type, ret_m in retrieval_metrics.items():
            val_log.update({f"valid/{ret_type}_{k}": round(v, 4) for k, v in ret_m.items()})

        TB_LOGGER.log_scalar_dict(val_log)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds."
                    f"itm_acc: {acc}. Retrieval res {retrieval_metrics}")


def start_training(cfg):
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    cfg.n_gpu = n_gpu
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

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
    train_loader, val_loader = setup_dataloaders(cfg, tokenizer)
    eval_loader = mk_video_ret_eval_dataloader(
        anno_path=cfg.val_datasets[0].txt,
        lmdb_dir=cfg.val_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer,
    )

    # compute the number of steps and update cfg
    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))

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
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 3
    running_loss = RunningMeter('train_loss')

    for step, batch in enumerate(InfiniteIterator(train_loader)):
        # forward pass
        del batch["caption_ids"]
        mini_batch = dict()
        for k, v in batch.items():
            if k != "visual_inputs":
                mini_batch[k] = v

        pool_method = cfg.score_agg_func
        # could be 1, where only a single clip is used
        num_clips = cfg.train_n_clips
        num_frm = cfg.num_frm
        # (B, T=num_clips*num_frm, C, H, W) --> (B, num_clips, num_frm, C, H, W)
        bsz = batch["visual_inputs"].shape[0]
        new_visual_shape = (bsz, num_clips, num_frm) + batch["visual_inputs"].shape[2:]
        visual_inputs = batch["visual_inputs"].view(*new_visual_shape)
        logits = []
        for clip_idx in range(num_clips):
            # (B, num_frm, C, H, W)
            mini_batch["visual_inputs"] = visual_inputs[:, clip_idx]
            mini_batch["n_examples_list"] = batch["n_examples_list"]
            outputs = forward_step(model, mini_batch, cfg)
            logits.append(outputs["logits"])
            # the losses are cross entropy and mse, no need to * num_labels

        logits = torch.stack(logits)  # (num_frm, B, 5)
        if pool_method == "mean":
            logits = logits.mean(0)  # (B, 5)
        elif pool_method == "max":
            logits = logits.max(0)[0]  # (B, 5)
        elif pool_method == "lse":
            logits = logits.permute(1, 0, 2).contiguous()  # (B, num_frm, 5), pooling will be done in CE
        else:
            raise ValueError(f"Invalid value for pool_method, "
                             f"got {pool_method}, expect one of [`mean`, `max`, `lse`]")

        if pool_method == "lse":
            out = torch.logsumexp(logits.view(logits.shape[0], -1), dim=-1, keepdim=True) \
                - torch.logsumexp(logits, dim=1)
            loss = torch.gather(out, -1, batch["labels"].view(-1, 1))
        else:
             _, loss = model.transformer.calc_loss(
                logits, batch["labels"], sample_size=len(batch["n_examples_list"]))
        loss = loss.mean()

        running_loss(loss.item())
        # backward pass
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

            # learning rate scheduling
            n_epoch = int(1. * total_train_batch_size * global_step
                          / total_n_examples)
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

            TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer),
                    cfg.grad_norm)
                TB_LOGGER.add_scalar(
                    "train/grad_norm", grad_norm, global_step)
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
                validate(
                    model, val_loader, eval_loader, cfg, global_step,
                    eval_filepath=cfg.val_datasets[0].txt)
                model_saver.save(step=global_step, model=model)
        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(
            model, val_loader, eval_loader, cfg, global_step,
            eval_filepath=cfg.val_datasets[0].txt)
        model_saver.save(step=global_step, model=model)


def get_retrieval_metric_from_bool_matrix(bool_matrix):
    """ Calc Recall@K, median rank and mean rank.
    Args:
        bool_matrix: np array of shape (#txt, #vid), np.bool,
            sorted row-wise from most similar to less similar.
            The GT position is marked as 1, while all the others are 0,
            each row will only have one 1.

    Returns:
        retrieval_metrics: dict(
            R1=.., R5=..., R10=..., MedR=..., MeanR=...
        )
    """
    num_row = bool_matrix.shape[0]  # #rows
    row_range, gt_ranks = np.where(bool_matrix == 1)
    assert np.allclose(row_range, np.arange(len(row_range))), \
        "each row should only a single GT"
    retrieval_metrics = dict(
        r1=100 * bool_matrix[:, 0].sum() / num_row,
        r5=100 * bool_matrix[:, :5].sum() / num_row,
        r10=100 * bool_matrix[:, :10].sum() / num_row,
        medianR=np.median(gt_ranks+1),  # convert to 1-indexed system instead of 0-indexed.
        meanR=np.mean(gt_ranks+1)
    )
    return retrieval_metrics


def get_retrieval_scores(score_matrix, gt_row2col_id_mapping, row_idx2id, col_id2idx):
    # rank scores
    score_matrix_sorted, indices_sorted = \
        torch.sort(score_matrix, dim=1, descending=True)  # (#txt, #vid)

    # build bool matrix, where the GT position is marked as 1, all the others are 0,
    num_row = len(score_matrix)
    gt_col_indices = torch.zeros(num_row, 1)
    for idx in range(num_row):
        gt_col_id = gt_row2col_id_mapping[row_idx2id[idx]]
        gt_col_indices[idx, 0] = col_id2idx[gt_col_id]

    bool_matrix = indices_sorted == gt_col_indices  # (#txt, #vid)
    retrieval_metrics = get_retrieval_metric_from_bool_matrix(bool_matrix.numpy())
    return retrieval_metrics


def eval_retrieval(vid_txt_score_dicts, gt_txt_id2vid_id, id2data):
    """
    Args:
        vid_txt_score_dicts: list(dict), each dict is dict(vid_id=..., txt_id=..., score=...)
        gt_txt_id2vid_id: dict, ground-truth {txt_id: vid_id}
        id2data: dict, {txt_id: single_example_data}

    Returns:

    """
    # group prediction by txt_id
    scores_group_by_txt_ids = defaultdict(list)
    for d in vid_txt_score_dicts:
        scores_group_by_txt_ids[d["txt_id"]].append(d)

    # clean duplicated videos
    _scores_group_by_txt_ids = defaultdict(list)
    for txt_id, txt_vid_pairs in scores_group_by_txt_ids.items():
        added_vid_ids = []
        for d in txt_vid_pairs:
            if d["vid_id"] not in added_vid_ids:
                _scores_group_by_txt_ids[txt_id].append(d)
                added_vid_ids.append(d["vid_id"])
    scores_group_by_txt_ids = _scores_group_by_txt_ids

    num_txt = len(scores_group_by_txt_ids)
    any_key = list(scores_group_by_txt_ids.keys())[0]
    vid_ids = [d["vid_id"] for d in scores_group_by_txt_ids[any_key]]
    num_vid = len(vid_ids)
    assert len(set(vid_ids)) == num_vid, "Each caption should be compared to each video only once."
    for k, v in scores_group_by_txt_ids.items():
        assert num_vid == len(v), "each captions should be compared with the same #videos."

    # row/col indices in the score matrix
    # *_id are the original ids, *_idx are the matrix indices
    txt_id2idx = {txt_id: idx for idx, txt_id in enumerate(scores_group_by_txt_ids)}
    vid_id2idx = {vid_id: idx for idx, vid_id in enumerate(vid_ids)}
    txt_idx2id = {v: k for k, v in txt_id2idx.items()}
    vid_idx2id = {v: k for k, v in vid_id2idx.items()}

    # build score (float32) and vid_id (str) matrix
    score_matrix = torch.zeros(num_txt, num_vid)
    for txt_id, preds in scores_group_by_txt_ids.items():
        txt_idx = txt_id2idx[txt_id]
        for p in preds:
            vid_idx = vid_id2idx[p["vid_id"]]
            score_matrix[txt_idx, vid_idx] = p["score"]

    # text to video retrieval, score_matrix--> (#txt, #vid)
    # given a text, retrieve most relevant videos
    t2v_retrieval_metrics = get_retrieval_scores(
        score_matrix, gt_txt_id2vid_id, txt_idx2id, vid_id2idx)
    # video to text retrieval, score_matrix--> (#vid, #txt)
    # given a video, retrieve most relevant videos
    score_matrix = score_matrix.transpose(0, 1)
    gt_vid_id2txt_id = {v: k for k, v in gt_txt_id2vid_id.items()}
    v2t_retrieval_metrics = get_retrieval_scores(
        score_matrix, gt_vid_id2txt_id, vid_idx2id, txt_id2idx)
    retrieval_metrics = dict(
        text2video=t2v_retrieval_metrics,
        video2text=v2t_retrieval_metrics
    )
    return retrieval_metrics


@torch.no_grad()
def inference_retrieval(model, val_loader, eval_file_path, cfg):
    model.eval()
    retrieval_res = []  # list(dict): dict(vid_id=..., txt_id=..., score=...)
    st = time.time()
    eval_bsz = cfg.inference_batch_size if cfg.do_inference else cfg.eval_retrieval_batch_size
    LOGGER.info(f"Evaluate retrieval #video per GPU: {len(val_loader)}")
    if hvd.rank() == 0:
        pbar = tqdm(total=len(val_loader), desc="eval")

    for batch in val_loader:
        # each batch contains 1 video and N (=1000) captions
        n_mini_batches = math.ceil(len(batch["caption_ids"]) / eval_bsz)
        vid_id = batch["vid_id"]
        for idx in range(n_mini_batches):
            # compile shared text part
            mini_batch = dict()
            for k in ["text_input_ids", "text_input_mask", "labels"]:
                if batch[k] is not None:
                    mini_batch[k] = batch[k][idx * eval_bsz:(idx + 1) * eval_bsz]
                else:
                    mini_batch[k] = None
            caption_ids = batch["caption_ids"][idx * eval_bsz:(idx + 1) * eval_bsz]
            # bsz = len(caption_ids)
            mini_batch["n_examples_list"] = [len(caption_ids)]

            # multi-frame test, scores across frames of the same video will be pooled together
            pool_method = cfg.score_agg_func
            # could be 1, where only a single clip is evaluated
            num_clips = cfg.inference_n_clips
            num_frm = cfg.num_frm
            # (B, T=num_clips*num_frm, C, H, W) --> (B, num_clips, num_frm, C, H, W)
            new_visual_shape = (1, num_clips, num_frm) + batch["visual_inputs"].shape[2:]
            visual_inputs = batch["visual_inputs"].view(*new_visual_shape)
            logits = []
            for clip_idx in range(num_clips):
                mini_batch["visual_inputs"] = visual_inputs[:, clip_idx]
                outputs = forward_step(model, mini_batch, cfg)
                logits.append(outputs["logits"].cpu())
            logits = torch.stack(logits)  # (num_frm, B, 1 or 2)
            if pool_method == "mean":
                logits = logits.mean(0)  # (B, 1 or 2)
            elif pool_method == "max":
                logits = logits.max(0)[0]  # (B, 1 or 2)
            elif pool_method == "lse":
                logits = logits.permute(1, 0, 2).contiguous()  # (B, num_frm, 5), pooling will be done in CE
                logits = torch.logsumexp(logits, dim=1)  # torch.exp alone might be too large and unstable
            else:
                raise ValueError(f"Invalid value for pool_method, "
                                 f"got {pool_method}, expect one of [`mean`, `max`, `lse`]")

            if logits.shape[1] == 2:
                probs = F.softmax(logits, dim=1)[:, 1].tolist()
            else:
                probs = torch.sigmoid(logits.squeeze()).tolist()  # B
            for cap_id, score in zip(caption_ids, probs):
                retrieval_res.append(dict(
                    vid_id=vid_id,
                    txt_id=cap_id,
                    score=round(score, 4)
                ))

        if hvd.rank() == 0:
            pbar.update(1)

    # ###### Saving with Horovod ####################
    # dummy sync
    _ = None
    all_gather_list(_)
    n_gpu = hvd.size()
    eval_dir = join(cfg.output_dir, f"results_{os.path.splitext(os.path.basename(eval_file_path))[0]}")
    os.makedirs(eval_dir, exist_ok=True)
    if n_gpu > 1:
        # with retrial, as azure blob fails occasionally.
        max_save_load_trial = 10
        save_trial = 0
        while save_trial < max_save_load_trial:
            try:
                LOGGER.info(f"Save results trial NO. {save_trial}")
                save_json(
                    retrieval_res,
                    join(eval_dir, f"tmp_results_rank{hvd.rank()}.json"))
                break
            except Exception as e:
                print(f"Saving exception: {e}")
                save_trial += 1

    # dummy sync
    _ = None
    all_gather_list(_)
    # join results
    if n_gpu > 1 and hvd.rank() == 0:
        retrieval_res = []
        for rk in range(n_gpu):
            retrieval_res.extend(load_json(
                join(eval_dir, f"tmp_results_rank{rk}.json")))
        LOGGER.info('results joined')

    if hvd.rank() == 0:
        retrieval_metrics = eval_retrieval(
            retrieval_res, val_loader.dataset.gt_cap_id2vid_id, val_loader.dataset.id2data)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds. scores: {retrieval_metrics}")
    else:
        retrieval_metrics = None

    model.train()
    return retrieval_res, retrieval_metrics


def start_inference(cfg):
    set_random_seed(cfg.seed)
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True

    inference_res_dir = join(
        cfg.output_dir,
        f"results_{os.path.splitext(os.path.basename(cfg.inference_txt_db))[0]}/"
        f"step_{cfg.inference_model_step}_{cfg.inference_n_clips}_{cfg.score_agg_func}"
    )

    if hvd.rank() == 0:
        os.makedirs(inference_res_dir, exist_ok=True)
        save_json(cfg, join(inference_res_dir, "raw_args.json"),
                  save_pretty=True)

    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    # overwrite cfg with stored_cfg,
    # but skip keys containing the keyword 'inference'
    stored_cfg_path = join(cfg.output_dir, "log/args.json")
    stored_cfg = edict(load_json(stored_cfg_path))
    for k, v in cfg.items():
        if k in stored_cfg and "inference" not in k and "output_dir" not in k:
            setattr(cfg, k, stored_cfg[k])

    # setup models
    cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    e2e_weights_path = join(
        cfg.output_dir, f"ckpt/model_step_{cfg.inference_model_step}.pt")
    if exists(e2e_weights_path):
        cfg.e2e_weights_path = e2e_weights_path
    else:
        cfg.bert_weights_path = join(
            f"{cfg.output_dir}/ckpt",
            f"transformer_step_{cfg.inference_model_step}.pt")
        cfg.cnn_weights_path = join(
            cfg.output_dir, f"ckpt/cnn_step_{cfg.inference_model_step}.pt")
    model = setup_model(cfg, device=device)
    model.eval()

    # FIXME separate scaling for each loss
    model = amp.initialize(
        model, enabled=cfg.fp16, opt_level='O2')

    global_step = 0
    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    cfg.data_ratio = 1.

    val_loader = mk_video_ret_eval_dataloader(
        anno_path=cfg.inference_txt_db,
        lmdb_dir=cfg.inference_img_db,
        cfg=cfg, tokenizer=tokenizer,
    )

    LOGGER.info(cfg)
    LOGGER.info("Starting inference...")
    LOGGER.info(f"***** Running inference with {n_gpu} GPUs *****")
    LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

    LOGGER.info(f'Step {global_step}: start validation')
    ret_results, ret_scores = inference_retrieval(
        model, val_loader, cfg.inference_txt_db, cfg)

    if hvd.rank() == 0:
        save_json(cfg, join(inference_res_dir, "merged_args.json"),
                  save_pretty=True)
        save_json(ret_results, join(inference_res_dir, "results.json"),
                  save_pretty=True)
        save_json(ret_scores, join(inference_res_dir, "scores.json"),
                  save_pretty=True)


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    input_cfg = shared_configs.get_video_retrieval_args()
    if input_cfg.do_inference:
        start_inference(input_cfg)
    else:
        start_training(input_cfg)
