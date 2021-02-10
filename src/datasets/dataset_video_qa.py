import torch
import random
import numpy as np
import copy
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import ClipBertBaseDataset


class ClipBertVideoQADataset(ClipBertBaseDataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """
    open_ended_qa_names = ["frameqa", "msrvtt_qa"]

    def __init__(self, task_type, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, ans2label=None,
                 ensemble_n_clips=1, return_label=True, is_train=True, random_sample_clips=True):
        super(ClipBertVideoQADataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.return_label = return_label
        self.is_train = is_train
        self.task_type = task_type
        self.ans2label = ans2label
        self.num_labels = len(ans2label)
        self.random_sample_clips = random_sample_clips
        self.label2ans = {v: k for k, v in ans2label.items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

    def __len__(self):
        return len(self.datalist)

    def _load_video_multi_clips_uniform(self, vid_id):
        """take multiple clips at fixed position"""
        vid_frm_array_list = []
        prev_clip = None
        video_max_pts = None
        for clip_idx in range(self.ensemble_n_clips):
            curr_clip, video_max_pts = self._load_video(
                vid_id, num_clips=self.ensemble_n_clips,
                clip_idx=clip_idx, safeguard_duration=True,
                video_max_pts=video_max_pts)
            if curr_clip is None:
                print("Copying prev clips as the current one is None")
                curr_clip = copy.deepcopy(prev_clip)
            else:
                prev_clip = curr_clip
            vid_frm_array_list.append(curr_clip)
        return torch.cat(vid_frm_array_list, dim=0)

    def _load_video_multi_clips_random(self, vid_id):
        """take multiple clips at random position"""
        vid_frm_array_list = []
        prev_clip = None
        for clip_idx in range(self.ensemble_n_clips):
            curr_clip, _ = self._load_video(
                vid_id, num_clips=None, clip_idx=None,
                safeguard_duration=False)
            if curr_clip is None:
                print("Copying prev clips as the current one is None")
                curr_clip = copy.deepcopy(prev_clip)
            else:
                prev_clip = curr_clip
            vid_frm_array_list.append(curr_clip)
        return None if any([e is None for e in vid_frm_array_list]) else torch.cat(vid_frm_array_list, dim=0)

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 3
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            if self.ensemble_n_clips > 1:
                # tensor (T*ensemble_n_clips, C, H, W), reshape as (T, ensemble_n_clips, C, H, W)
                if self.is_train and self.random_sample_clips:
                    vid_frm_array = self._load_video_multi_clips_random(vid_id)
                else:
                    vid_frm_array = self._load_video_multi_clips_uniform(vid_id)
            else:
                if self.is_train and self.random_sample_clips:
                    vid_frm_array, _ = self._load_video(vid_id)  # tensor (T, C, H, W)
                else:
                    vid_frm_array, _ = self._load_video(vid_id, num_clips=1, clip_idx=0)  # tensor (T, C, H, W)
            # vid_frm_array = torch.zeros_like(vid_frm_array)
            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            examples = [self._get_single_example(e) for e in examples]
            return dict(
                vid=vid_frm_array,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"]
        )
        if self.task_type in ["action", "transition"]:
            example["options_str_list"] = data["options"]
        elif self.task_type in self.open_ended_qa_names:
            if self.return_label:
                example["label"] = self.ans2label[example["label"]]
        if not self.return_label:
            example["label"] = None
        return example

    def evaluate_tgif_qa(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "question_id": int,
                    "answer": int or float, either answer_idx (int)
                }
        Returns:
            TGIF-QA score
        """
        preds = []
        gts = []
        # for frameQA
        answer_types = []
        answer_type2idx = dict(
            frameqa={"object": 0, "number": 1, "color": 2, "location": 3},
            msrvtt_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
        )

        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
        if self.task_type in self.open_ended_qa_names:  # convert ans_idx, int --> str
            qid2pred_ans = {k: self.label2ans[v] for k, v in qid2pred_ans.items()}

        for qid, pred_ans in qid2pred_ans.items():
            preds.append(pred_ans)

            gt_data = self.qid2data[qid]
            gt_ans = gt_data["answer"]
            if self.task_type in self.open_ended_qa_names:
                answer_types.append(answer_type2idx[self.task_type][gt_data["answer_type"]])
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        metrics["overall_acc"] = float(np.mean(preds == gts))
        if self.task_type in self.open_ended_qa_names:
            answer_types = np.array(answer_types)
            ratios = dict()
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                answer_type_mask = answer_types == ans_type_idx
                answer_type_corrects = (
                        preds[answer_type_mask] == gts[answer_type_mask])
                metrics[f"{ans_type}_acc"] = float(
                    np.mean(answer_type_corrects)) if len(answer_type_corrects) != 0 else 0
                ratios[f"{ans_type}_ratio"] = [
                    1. * len(answer_type_corrects) / len(answer_types),
                    len(answer_type_corrects)]
            metrics["ratios"] = ratios
        return metrics


class VideoQACollator(object):
    def __init__(self, tokenizer, max_length=20, task_type="action", n_options=5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.n_options = n_options

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in ["action", "transition"]:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )  # (B * n_options, )
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        return dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,
            labels=labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
