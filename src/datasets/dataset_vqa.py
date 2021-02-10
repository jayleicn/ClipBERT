import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.dataset_base import ClipBertBaseDataset, img_collate


class ClipBertVQADataset(ClipBertBaseDataset):
    """ This should work for both train and test (where labels are not available).
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict {
            "type": "image",
            "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
            "text": "A plate of food and a beverage are on a table.",
            "labels": {"down": 1, "at table": 0.3, "skateboard": 0.3, "table": 0.3}
            "answer_type": "other"
            "question_id": 262148000
            }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir, fps=3,
                 max_img_size=1000, max_txt_len=20, ans2label=None):
        super(ClipBertVQADataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, fps=fps,
            max_img_size=max_img_size, max_txt_len=max_txt_len,
            )  # init its parent class

        self.ans2label = ans2label
        self.num_labels = len(ans2label)
        self.label2ans = {v: k for k, v in ans2label.items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        img_id, examples = self.datalist[index]  # one image with multiple examples
        img_array = self._load_img(img_id)  # tensor
        examples = [self._get_single_example(e) for e in examples]
        return dict(
            img=img_array,
            examples=examples,
            n_examples=len(examples)  # used to create image feature copies.
        )

    def _get_single_example(self, data):
        return dict(
            text_str=data["txt"],
            question_id=data["question_id"],
            labels=self._get_vqa_targets(
                data["labels"]) if "labels" in data else None
        )

    def _get_vqa_targets(self, ans2score_dict):
        """
        Args:
            ans2score_dict: {"table": 0.9, "picnic table": 1,
                             "skateboard": 0.3}
        Returns:
            A 1D tensor
        """
        targets = torch.zeros(self.num_labels)
        raw_answers = list(ans2score_dict.keys())
        scores = [ans2score_dict[k] for k in raw_answers]
        labels = [self.ans2label[ans] for ans in raw_answers]
        targets.scatter_(
            0, torch.tensor(labels).long(),
            torch.tensor(scores).float())
        return targets

    def evaluate_vqa(self, results):
        """
        Args:
            results: list(dict), in accordance with VQA online submission format
              each dict is
                {
                    "question_id": int,
                    "answer": str
                }
        Returns:
            VQA score
        """
        scores = []
        answer_types = []
        answer_type2idx = {"yes/no": 0, "number": 1, "other": 2}
        for d in results:
            qid = d["question_id"]
            ans = d["answer"]
            raw_data = self.qid2data[qid]
            labels = raw_data["labels"]
            if ans in labels:
                scores.append(labels[ans])
            else:
                scores.append(0.)
            answer_types.append(answer_type2idx[raw_data["answer_type"]])
        metrics = dict()
        scores = np.array(scores)
        metrics["overall_acc"] = float(np.mean(scores))
        answer_types = np.array(answer_types)
        ratios = dict()
        for ans_type, ans_type_idx in answer_type2idx.items():
            answer_type_mask = answer_types == ans_type_idx
            answer_type_scores = scores[answer_type_mask]
            metrics[f"{ans_type}_acc"] = float(np.mean(answer_type_scores))
            ratios[f"{ans_type}_ratio"] = [
                1. * len(answer_type_scores) / len(scores),
                len(answer_type_scores)]
        metrics["ratios"] = ratios
        return metrics


class VQACollator(object):
    def __init__(self, tokenizer, max_length=20):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch):
        if isinstance(batch[0]["img"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        visual_inputs = v_collate([d["img"] for d in batch])  # (B, #frm=1 or T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        batch_enc = self.tokenizer.batch_encode_plus(
            [d["text_str"] for d in text_examples],
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)
        labels = default_collate(
            [d["labels"] for d in text_examples]) \
            if text_examples[0]["labels"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        return dict(
            visual_inputs=visual_inputs,  # (B, #frm=1 or T, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,
            labels=labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
