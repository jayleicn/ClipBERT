"""
Modified from UNITER code
"""
import os
import sys
import json
import argparse

from easydict import EasyDict as edict


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object, enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class SharedConfigs(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self, desc="shared config for pretraining and finetuning"):
        parser = argparse.ArgumentParser(description=desc)
        # debug parameters
        parser.add_argument(
            "--debug", type=int, choices=[0, 1], default=0,
            help="debug mode, output extra info & break all loops."
                 "0: disable, 1 enable")
        parser.add_argument(
            "--data_ratio", type=float, default=1.0,
            help="portion of train/val exampels to use,"
                 "e.g., overfit a small set of data")

        # Required parameters
        parser.add_argument(
            "--model_config", type=str,
            help="path to model structure config json")
        parser.add_argument(
            "--tokenizer_dir", type=str, help="path to tokenizer dir")
        parser.add_argument(
            "--output_dir", type=str,
            help="dir to store model checkpoints & training meta.")

        # data preprocessing parameters
        parser.add_argument(
            "--max_txt_len", type=int, default=20, help="max text #tokens ")
        parser.add_argument(
            "--max_img_size", type=int, default=448,
            help="max image longer side size, shorter side will be padded with zeros")
        parser.add_argument(
            "--img_pixel_mean", type=float, default=None,
            nargs=3, help="image pixel mean")
        parser.add_argument(
            "--img_pixel_std", type=float, default=None,
            nargs=3, help="image pixel std")
        parser.add_argument(
            "--img_input_format", type=str, default="BGR",
            choices=["BGR", "RGB"], help="image input format is BGR for detectron2")
        parser.add_argument(
            "--max_n_example_per_group", type=int, default=2,
            help="max #examples (e.g., captions) paired with each image/video in an input group."
                 "1: each image is paired with a single sent., equivalent to sample by sent.;"
                 "X (X>1): each image can be paired with a maximum of X sent.; X>1 can be used "
                 "to reduce image processing time, including basic transform (resize, etc) and CNN encoding"
        )
        # video specific parameters
        parser.add_argument("--fps", type=int, default=1, help="video frame rate to use")
        parser.add_argument("--num_frm", type=int, default=3,
                            help="#frames to use per clip -- we first sample a clip from a video, "
                                 "then uniformly sample num_frm from the clip. The length of the clip "
                                 "will be fps * num_frm")
        parser.add_argument("--frm_sampling_strategy", type=str, default="rand",
                            choices=["rand", "uniform", "start", "middle", "end"],
                            help="see src.datasets.dataset_base.extract_frames_from_video_binary for details")

        # MLL training settings
        parser.add_argument("--train_n_clips", type=int, default=3,
                            help="#clips to sample from each video for MIL training")
        parser.add_argument("--score_agg_func", type=str, default="mean",
                            choices=["mean", "max", "lse"],
                            help="score (from multiple clips) aggregation function, lse = LogSumExp")
        parser.add_argument("--random_sample_clips", type=int, default=1, choices=[0, 1],
                            help="randomly sample clips for training, otherwise use uniformly sampled clips.")

        # training parameters
        parser.add_argument(
            "--train_batch_size", default=128, type=int,
            help="Single-GPU batch size for training for Horovod.")
        parser.add_argument(
            "--val_batch_size", default=128, type=int,
            help="Single-GPU batch size for validation for Horovod.")
        parser.add_argument(
            "--gradient_accumulation_steps", type=int, default=1,
            help="#updates steps to accumulate before performing a backward/update pass."
                 "Used to simulate larger batch size training. The simulated batch size "
                 "is train_batch_size * gradient_accumulation_steps for a single GPU.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="initial learning rate.")
        parser.add_argument(
            "--num_valid", default=20, type=int,
            help="Run validation X times during training and checkpoint.")
        parser.add_argument(
            "--min_valid_steps", default=100, type=int,
            help="minimum #steps between two validation runs")
        parser.add_argument(
            "--save_steps_ratio", default=0.01, type=float,
            help="save every 0.01*global steps to resume after preemption,"
                 "not used for checkpointing.")
        parser.add_argument("--num_train_epochs", default=10, type=int,
                            help="Total #training epochs.")
        parser.add_argument("--optim", default="adamw",
                            choices=["adam", "adamax", "adamw"],
                            help="optimizer")
        parser.add_argument("--betas", default=[0.9, 0.98],
                            nargs=2, help="beta for adam optimizer")
        parser.add_argument("--decay", default="linear",
                            choices=["linear", "invsqrt"],
                            help="learning rate decay method")
        parser.add_argument("--dropout", default=0.1, type=float,
                            help="tune dropout regularization")
        parser.add_argument("--weight_decay", default=1e-3, type=float,
                            help="weight decay (L2) regularization")
        parser.add_argument("--grad_norm", default=2.0, type=float,
                            help="gradient clipping (-1 for no clipping)")
        parser.add_argument(
            "--warmup_ratio", default=0.1, type=float,
            help="to perform linear learning rate warmup for. (invsqrt decay)")
        parser.add_argument("--transformer_lr_mul", default=1.0, type=float,
                            help="lr_mul for transformer")
        parser.add_argument(
            "--transformer_lr_mul_prefix", default="", type=str,
            help="lr_mul param prefix for transformer")
        parser.add_argument("--step_decay_epochs", type=int,
                            nargs="+", help="cnn multi_step decay epochs")
        # CNN parameters
        parser.add_argument("--cnn_optim", default="adamw", type=str,
                            choices=["adam", "adamax", "adamw", "sgd"],
                            help="optimizer for CNN")
        parser.add_argument("--cnn_learning_rate", default=5e-5, type=float,
                            help="learning rate for CNN")
        parser.add_argument("--cnn_weight_decay", default=1e-3, type=float,
                            help="weight decay for CNN")
        parser.add_argument("--cnn_sgd_momentum", default=0.9, type=float,
                            help="momentum for SGD")
        parser.add_argument("--cnn_lr_mul", default=1.0, type=float,
                            help="lr_mul for CNN")
        parser.add_argument(
            "--cnn_lr_mul_prefix", default="grid_encoder", type=str,
            help="lr_mul param prefix for CNN")
        parser.add_argument("--cnn_lr_decay", default="linear",
                            choices=["linear", "invsqrt", "multi_step",
                                     "constant"],
                            help="learning rate decay method")
        parser.add_argument("--cnn_step_decay_epochs", type=int,
                            nargs="+", help="cnn multi_step decay epochs")
        parser.add_argument(
            "--freeze_cnn", default=0, choices=[0, 1], type=int,
            help="freeze CNN by setting the requires_grad=False for CNN parameters.")

        # model arch
        parser.add_argument(
            "--detectron2_model_cfg", type=str, default="",
            help="path to detectron2 model cfg yaml")

        # checkpoint
        parser.add_argument("--e2e_weights_path", type=str,
                            help="path to e2e model weights")
        parser.add_argument("--detectron2_weights_path", type=str,
                            help="path to detectron2 weights, only use for pretraining")
        parser.add_argument("--bert_weights_path", type=str,
                            help="path to BERT weights, only use for pretraining")

        # inference only, please include substring `inference'
        # in the option to avoid been overwrite by loaded options,
        # see start_inference() in run_vqa_w_hvd.py
        parser.add_argument("--inference_model_step", default=-1, type=int,
                            help="pretrained model checkpoint step")
        parser.add_argument(
            "--do_inference", default=0, type=int, choices=[0, 1],
            help="perform inference run. 0: disable, 1 enable")
        parser.add_argument(
            "--inference_split", default="val",
            help="For val, the data should have ground-truth associated it."
                 "For test*, the data comes with no ground-truth.")
        parser.add_argument("--inference_txt_db", type=str,
                            help="path to txt_db file for inference")
        parser.add_argument("--inference_img_db", type=str,
                            help="path to img_db file for inference")
        parser.add_argument("--inference_batch_size", type=int, default=64,
                            help="single-GPU batch size for inference")
        parser.add_argument("--inference_n_clips", type=int, default=1,
                            help="uniformly sample `ensemble_n_clips` clips, "
                                 "each contains `num_frm` frames. When it == 1, "
                                 "use the frm_sampling_strategy to sample num_frm frames."
                                 "When it > 1, ignore frm_sampling_strategy, "
                                 "uniformly sample N clips, each clips num_frm frames.")

        # device parameters
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument(
            "--fp16", type=int, choices=[0, 1], default=0,
            help="Use 16-bit float precision instead of 32-bit."
                 "0: disable, 1 enable")
        parser.add_argument("--n_workers", type=int, default=4,
                            help="#workers for data loading")
        parser.add_argument("--pin_mem", type=int, choices=[0, 1], default=1,
                            help="pin memory. 0: disable, 1 enable")

        # can use config files, will only overwrite unset parameters
        parser.add_argument("--config", help="JSON config files")
        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        args = parse_with_config(parsed_args)

        # convert to all [0, 1] options to bool, including these task specific ones
        zero_one_options = [
            "fp16", "pin_mem", "use_itm", "use_mlm", "debug", "freeze_cnn",
            "do_inference",
        ]
        for option in zero_one_options:
            if hasattr(args, option):
                setattr(args, option, bool(getattr(args, option)))

        # basic checks
        # This is handled at TrainingRestorer
        # if exists(args.output_dir) and os.listdir(args.output_dir):
        #     raise ValueError(f"Output directory ({args.output_dir}) "
        #                      f"already exists and is not empty.")
        if args.cnn_step_decay_epochs and args.cnn_lr_decay != "multi_step":
            Warning(
                f"--cnn_step_decay_epochs set to {args.cnn_step_decay_epochs}"
                f"but will not be effective, as --cnn_lr_decay set to be {args.cnn_lr_decay}")
        if args.step_decay_epochs and args.decay != "multi_step":
            Warning(
                f"--step_decay_epochs epochs set to {args.step_decay_epochs}"
                f"but will not be effective, as --decay set to be {args.decay}")

        assert args.gradient_accumulation_steps >= 1, \
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps} "

        assert 1 >= args.data_ratio > 0, \
            f"--data_ratio should be [1.0, 0), but get {args.data_ratio}"

        assert args.max_img_size > 0, "max_img_size need to be set > 0"

        if args.score_agg_func == "lse":
            assert args.loss_type == "ce",\
                f"lse method can only work with ce loss, not {args.loss_type}"
        return args

    def get_pretraining_args(self):
        self.parser.add_argument(
            "--itm_neg_prob", default=0.5, type=float,
            help="probability to make negative examples in ITM training")
        self.parser.add_argument(
            "--use_itm", type=int, choices=[0, 1], default=0,
            help="enable itm loss. 0: disable, 1 enable")
        self.parser.add_argument(
            "--use_mlm", type=int, choices=[0, 1], default=0,
            help="enable mlm loss. 0: disable, 1 enable")
        self.parser.add_argument(
            "--pixel_random_sampling_size", type=int, default=0,
            help="use pixel_random_sampling at pre-training, "
                 "0: disable, positive int: enable. In Pixel-BERT, it is 100")
        args = self.parse_args()
        return args

    def get_video_retrieval_args(self):
        self.parser.add_argument(
            "--itm_neg_size", default=1, type=int,
            help="#negative captions to sample for each image")
        self.parser.add_argument("--classifier", type=str, default="mlp",
                                 choices=["mlp", "linear"],
                                 help="classifier type")
        self.parser.add_argument(
            "--cls_hidden_scale", type=int, default=2,
            help="scaler of the intermediate linear layer dimension for mlp classifier")
        self.parser.add_argument(
            "--margin", default=0.2, type=float,
            help="ranking loss margin")
        self.parser.add_argument("--loss_type", type=str, default="ce",
                                 choices=["ce", "rank"],
                                 help="loss type")
        self.parser.add_argument("--eval_retrieval_batch_size", type=int, default=256,
                                 help="batch size for retrieval, since each batch will only have one image, "
                                      "retrieval allows larger batch size")

        args = self.parse_args()
        args.num_labels = 1 if args.loss_type == "rank" else 2
        return args

    def get_vqa_args(self):
        self.parser.add_argument("--ans2label_path", type=str,
                                 help="path to {answer: label} file")
        self.parser.add_argument("--loss_type", type=str, default="bce",
                                 help="loss type")
        self.parser.add_argument("--classifier", type=str, default="mlp",
                                 choices=["mlp", "linear"],
                                 help="classifier type")
        self.parser.add_argument(
            "--cls_hidden_scale", type=int, default=2,
            help="scaler of the intermediate linear layer dimension for mlp classifier")
        self.parser.add_argument("--num_labels", type=int, default=3129,
                                 help="#labels/output-dim for classifier")
        return self.parse_args()

    def get_video_qa_args(self):
        self.parser.add_argument(
            "--task", type=str,
            choices=["action", "transition", "frameqa", "msrvtt_qa"],
            help="TGIF-QA tasks and MSRVTT-QA")
        self.parser.add_argument("--loss_type", type=str, default="ce",
                                 help="loss type, will be overwritten later")
        self.parser.add_argument("--classifier", type=str, default="mlp",
                                 choices=["mlp", "linear"],
                                 help="classifier type")
        self.parser.add_argument(
            "--cls_hidden_scale", type=int, default=2,
            help="scaler of the intermediate linear layer dimension for mlp classifier")
        # for frameQA msrvtt_qa
        self.parser.add_argument("--ans2label_path", type=str, default=None,
                                 help="path to {answer: label} file")

        # manually setup config by task type
        args = self.parse_args()
        if args.max_n_example_per_group != 1:
            Warning(f"For TGIF-QA, most GIF is only paired with a single example, no need to"
                    f"use max_n_example_per_group={args.max_n_example_per_group}"
                    f"larger than 1. Automatically reset to 1.")
            args.max_n_example_per_group = 1
        if os.path.exists(args.ans2label_path):
            num_answers = len(json.load(open(args.ans2label_path, "r")))
        else:
            num_answers = 0

        if args.task in ["action", "transition"]:
            args.num_labels = 5
            args.loss_type = "ce"
        elif args.task == "frameqa":
            args.num_labels = max(num_answers, 1540)
            args.loss_type = "ce"
        elif args.task == "msrvtt_qa":
            args.num_labels = max(num_answers, 1500)
            args.loss_type = "ce"
        else:
            raise NotImplementedError
        return args


shared_configs = SharedConfigs()
