import torch
import os
import time
import random
import math
from transformers import BertConfig, BertTokenizerFast
from src.modeling.modeling import ClipBertForSequenceClassification
from src.modeling.e2e_model import ClipBert

from src.datasets.dataset_vqa import ClipBertVQADataset, VQACollator
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json,
    get_rounded_percentage, flat_list_of_lists)
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.utils import setup_e2e_optimizer
from src.optimization.sched import get_lr_sched

from tqdm import tqdm
from os.path import join, exists
from easydict import EasyDict as edict
from apex import amp
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from src.utils.distributed import all_gather_list
from collections import defaultdict


def mk_vqa_dataloader(anno_path, img_lmdb_dir, cfg, tokenizer, is_train=True):
    """
    Returns:
        list(dict), each dict is
        {
            "filepath": str,
            "txt": str,
        }
    """
    if isinstance(anno_path, str):
        raw_datalist = load_jsonl(anno_path)
    else:
        raw_datalist = flat_list_of_lists([load_jsonl(p) for p in anno_path])

    if cfg.data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]

    datalist = []
    for raw_d in raw_datalist:
        d = dict(
            txt=raw_d["question"],
            img_id=raw_d["image_id"],
            question_id=raw_d["question_id"],
        )
        if "labels" in raw_d:  # deal with test sets
            d["labels"] = raw_d["labels"]
        if "answer_type" in raw_d:
            d["answer_type"] = raw_d["answer_type"]
        datalist.append(d)

    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["img_id"]].append(d)

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval
        is_train=is_train,
        example_unique_key="question_id"
    )

    ans2label = load_json(cfg.ans2label_path)
    dataset = ClipBertVQADataset(datalist=group_datalist,
                                 tokenizer=tokenizer,
                                 img_lmdb_dir=img_lmdb_dir,
                                 ans2label=ans2label,
                                 max_img_size=cfg.max_img_size,
                                 max_txt_len=cfg.max_txt_len)
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    vqa_collator = VQACollator(tokenizer=tokenizer,
                               max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vqa_collator.collate_batch)
    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loader = mk_vqa_dataloader(
        anno_path=cfg.train_datasets[0].txt,
        img_lmdb_dir=cfg.train_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=True
    )
    val_loader = mk_vqa_dataloader(
        anno_path=cfg.val_datasets[0].txt,
        img_lmdb_dir=cfg.val_datasets[0].img,
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
    add_attr_list = ["num_labels", "classifier",
                     "cls_hidden_scale", "loss_type"]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = ClipBert(
        model_cfg, input_format=cfg.img_input_format,
        detectron2_model_cfg=cfg.detectron2_model_cfg,
        transformer_cls=ClipBertForSequenceClassification)
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


def forward_step(model, batch):
    """shared for training and validation"""
    question_ids = batch["question_ids"]
    del batch["question_ids"]
    outputs = model(batch)  # dict
    return outputs, question_ids


@torch.no_grad()
def validate(model, val_loader, cfg, train_global_step, eval_score=True):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    model.eval()
    loss = 0.
    n_ex = 0
    vqa_results = []
    st = time.time()
    debug_step = 5
    for val_step, batch in enumerate(val_loader):
        # forward pass
        outputs, question_ids = forward_step(model, batch)

        loss += outputs["loss"].sum().item() if isinstance(
            outputs["loss"], torch.Tensor) else 0
        n_ex += len(question_ids)
        pred_labels = outputs["logits"].max(dim=-1)[1].data.tolist()
        for qid, pred_label in zip(question_ids, pred_labels):
            vqa_results.append(dict(
                question_id=qid,
                answer=val_loader.dataset.label2ans[pred_label]
            ))

        if cfg.debug and val_step >= debug_step:
            break

    if cfg.debug:
        LOGGER.info(vqa_results[:10])
    n_ex_per_rank = all_gather_list(n_ex)
    loss = sum(all_gather_list(loss))
    n_ex = sum(all_gather_list(n_ex))
    val_log = {'valid/loss': float(loss / n_ex)}
    if eval_score:
        LOGGER.info(f"Evaluate VQA scores for {len(vqa_results)} vqa_results,"
                    f"3 examples here: {vqa_results[:3]}")
        vqa_scores = val_loader.dataset.evaluate_vqa(vqa_results)

        # Gather scores
        scores_per_rank = all_gather_list(vqa_scores)
        gathered_scores = {}
        gathered_ratios = {
            k: [0, 0] for k, _ in scores_per_rank[0]["ratios"].items()}
        # Gather ratios
        for rank_id in range(len(n_ex_per_rank)):
            current_ratios = scores_per_rank[rank_id]["ratios"]
            for k, v in current_ratios.items():
                gathered_ratios[k][1] += v[1]
        for k, v in gathered_ratios.items():
            gathered_ratios[k][0] = get_rounded_percentage(1. * v[1] / n_ex)

        # FIXME: Gather scores become complicated due to np.mean and dict format.
        for scores_k, _ in vqa_scores.items():
            if "ratio" in scores_k:
                continue
            gathered_v = 0
            for rank_id, n in enumerate(n_ex_per_rank):
                if "overall" in scores_k:
                    curr_acc = scores_per_rank[rank_id][scores_k] * n
                else:
                    curr_n_ex = scores_per_rank[
                            rank_id]["ratios"][
                                scores_k.replace("acc", "ratio")][1]
                    curr_acc = scores_per_rank[rank_id][
                        scores_k] * curr_n_ex
                gathered_v += curr_acc
            if "overall" in scores_k:
                gathered_v = gathered_v * 1. / n_ex
            else:
                gathered_v = gathered_v * 1. / gathered_ratios[
                    scores_k.replace("acc", "ratio")][1]
            gathered_scores[scores_k] = get_rounded_percentage(gathered_v)
        gathered_scores["ratios"] = gathered_ratios

        for k, v in gathered_scores.items():
            if "ratio" not in k:
                val_log[f'valid/{k}'] = v
    else:
        LOGGER.info("Seems you are doing inference on test set,"
                    "no scores are calculated.")
        gathered_scores = 0

    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds."
                f"{gathered_scores}")
    model.train()
    return vqa_results


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
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
        pbar = tqdm(total=cfg.num_train_steps)
    else:
        LOGGER.disabled = True
        model_saver = NoOp()
        restorer = NoOp()
        pbar = NoOp()

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
        outputs, question_ids = forward_step(model, batch)
        loss = outputs["loss"].mean()
        loss = loss.float() * cfg.num_labels
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
            TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)

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
                vqa_results = validate(
                    model, val_loader, cfg, global_step)
                model_saver.save(step=global_step, model=model)
        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        vqa_results = validate(
            model, val_loader, cfg, global_step)
        model_saver.save(step=global_step, model=model)


def start_inference(cfg):
    set_random_seed(cfg.seed)
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True

    inference_res_dir = join(
        cfg.output_dir, f"results_{cfg.inference_split}"
        f"step_{cfg.inference_model_step}")
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
        if (k in stored_cfg and "inference" not in k
                and k != "output_dir"):
            value = stored_cfg[k]
            # FIXME hardcode changes
            if isinstance(value, str) and value.startswith("/data"):
                value = value.replace("/data", "/storage")
            setattr(cfg, k, value)

    # setup models
    cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    cfg.detectron2_model_cfg = join(
        cfg.output_dir, "log/detectron2_model_cfg.yaml")
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
    val_loader = mk_vqa_dataloader(
        anno_path=cfg.inference_txt_db,
        img_lmdb_dir=cfg.inference_img_db,
        cfg=cfg, tokenizer=tokenizer, is_train=False
    )
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    val_loader = PrefetchLoader(val_loader, img_norm)

    LOGGER.info(cfg)
    LOGGER.info("Starting inference...")
    LOGGER.info(f"***** Running inference with {n_gpu} GPUs *****")
    LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

    LOGGER.info(f'Step {global_step}: start validation')
    vqa_results = validate(
        model, val_loader, cfg, global_step,
        eval_score=cfg.inference_split == "val")

    if hvd.rank() == 0:
        save_json(cfg, join(inference_res_dir, "merged_args.json"),
                  save_pretty=True)

    # ###### Saving with Horovod ####################
    # dummy sync
    _ = None
    all_gather_list(_)
    if n_gpu > 1:
        # with retrial, as azure blob fails occasionally.
        max_save_load_trial = 10
        save_trial = 0
        while save_trial < max_save_load_trial:
            try:
                LOGGER.info(f"Save results trial NO. {save_trial}")
                save_json(
                    vqa_results,
                    join(inference_res_dir, f"results_rank{hvd.rank()}.json"))
                break
            except Exception:
                save_trial += 1
    # dummy sync
    _ = None
    all_gather_list(_)
    # join results
    if n_gpu > 1 and hvd.rank() == 0:
        vqa_results = []
        for rk in range(n_gpu):
            vqa_results.extend(load_json(
                join(inference_res_dir, f"results_rank{rk}.json")))
        LOGGER.info('results joined')

    if hvd.rank() == 0:
        save_json(
            vqa_results,
            join(inference_res_dir, "results_all.json"))
        LOGGER.info('all results written')


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    input_cfg = shared_configs.get_vqa_args()
    if input_cfg.do_inference:
        start_inference(input_cfg)
    else:
        start_training(input_cfg)
