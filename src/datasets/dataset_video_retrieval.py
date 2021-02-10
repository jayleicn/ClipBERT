import random
import copy
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import ClipBertBaseDataset


class ClipBertVideoRetrievalDataset(ClipBertBaseDataset):
    """ This should work for both train and test (where labels are not available).
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    random_sample_clips: bool, whether using randomly sampled N clips or always use uniformly sampled N clips
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=40, itm_neg_size=1,
                 ensemble_n_clips=1, random_sample_clips=True):
        super(ClipBertVideoRetrievalDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.num_labels = 2
        self.itm_neg_size = itm_neg_size
        self.random_sample_clips = random_sample_clips
        self.id2data = {
            d["id"]: d for group in datalist for d in group[1]}

    def __len__(self):
        return len(self.datalist)

    def _load_video_multi_clips_random(self, vid_id):
        """take multiple clips at fixed position"""
        vid_frm_array_list = []
        for clip_idx in range(self.ensemble_n_clips):
            frames, _ = self._load_video(vid_id, num_clips=None, clip_idx=None)
            vid_frm_array_list.append(frames)
        return None if any([e is None for e in vid_frm_array_list]) else torch.cat(vid_frm_array_list, dim=0)

    def _load_video_multi_clips_uniform(self, vid_id):
        vid_frm_array_list = []
        video_max_pts = None
        for clip_idx in range(self.ensemble_n_clips):
            frames, video_max_pts = self._load_video(
                vid_id,
                num_clips=self.ensemble_n_clips,
                clip_idx=clip_idx, safeguard_duration=True,
                video_max_pts=video_max_pts)
            vid_frm_array_list.append(frames)
        return None if any([e is None for e in vid_frm_array_list]) else torch.cat(vid_frm_array_list, dim=0)

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 3
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            if self.ensemble_n_clips > 1:
                if self.random_sample_clips:
                    vid_frm_array = self._load_video_multi_clips_random(vid_id)
                else:
                    vid_frm_array = self._load_video_multi_clips_uniform(vid_id)
            else:
                if self.random_sample_clips:
                    vid_frm_array, _ = self._load_video(vid_id)  # tensor (T, C, H, W)
                else:
                    vid_frm_array, _ = self._load_video(vid_id, num_clips=1, clip_idx=0)  # tensor (T, C, H, W)

            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            sampled_examples = []
            for e in examples:
                s = self._get_single_example(e, index)
                if isinstance(s, dict):
                    sampled_examples.append(s)
                else:
                    sampled_examples.extend(s)
            return dict(
                vid=vid_frm_array,
                examples=sampled_examples,
                n_examples=len(sampled_examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(
             f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data, index):
        examples = []

        text_str = data["txt"]
        itm_label = 1  # positive pair
        examples.append(dict(
            text_str=text_str,
            itm_label=itm_label
        ))
        count = 0
        while self.itm_neg_size > count:
            text_str = self._get_random_negative_caption(index)
            itm_label = 0  # negative pair
            examples.append(dict(
                text_str=text_str,
                itm_label=itm_label
            ))
            count += 1
        return examples

    def _get_random_negative_caption(self, gt_index):
        gt_img_id, _ = self.datalist[gt_index]
        neg_img_id = gt_img_id
        while neg_img_id == gt_img_id:
            neg_index = int(random.random() * len(self))
            neg_img_id, neg_examples = self.datalist[neg_index]
        neg_data = neg_examples[int(random.random() * len(neg_examples))]
        return neg_data["txt"]


class VideoRetrievalCollator(object):
    def __init__(self, tokenizer, max_length=40):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        text_str_list = [d["text_str"] for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        if "itm_label" in text_examples[0]:
            itm_labels = default_collate(
                [d["itm_label"] for d in text_examples])  # (B, )
        else:
            itm_labels = None

        if "id" in text_examples[0]:
            caption_ids = [d["id"] for d in text_examples]  # (B, )
        else:
            caption_ids = None
        collated_batch = dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            caption_ids=caption_ids,  # list(int), example ids,
            labels=itm_labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
        if "vid_id" in batch[0] and len(batch) == 1:
            collated_batch["vid_id"] = batch[0]["vid_id"]
        return collated_batch


class ClipBertVideoRetrievalEvalDataset(ClipBertBaseDataset):
    """ Sample by video/image, calculate scores between each video with all the text
    and loop through all the videos. Each batch will only contain a single video,
    but multiple text.

    datalist: list(dict), each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=40, ensemble_n_clips=1):
        self.ensemble_n_clips = ensemble_n_clips
        super(ClipBertVideoRetrievalEvalDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        # id is unique id per caption/example
        for i, d in enumerate(self.datalist):
            assert i == d["id"]
        self.gt_cap_id2vid_id = {d["id"]: d["vid_id"] for d in datalist}
        self.cap_id2data = {d["id"]: d for d in datalist}
        self.batches = self._prepare_batches_by_video()
        self.id2data = {d["id"]: d for d in self.datalist}

    def __len__(self):
        return len(self.batches)

    def _load_video_multi_clips_uniform(self, vid_id):
        vid_frm_array_list = []
        video_max_pts = None
        for clip_idx in range(self.ensemble_n_clips):
            frames, video_max_pts = self._load_video(
                vid_id,
                num_clips=self.ensemble_n_clips,
                clip_idx=clip_idx, safeguard_duration=True,
                video_max_pts=video_max_pts)
            vid_frm_array_list.append(frames)
        return torch.cat(vid_frm_array_list, dim=0)

    def __getitem__(self, index):
        # skip error videos:
        batch = self.batches[index]  # one video with multiple examples
        vid_id = batch["vid_id"]
        if self.ensemble_n_clips > 1:
            # tensor (T*ensemble_n_clips, C, H, W), reshape as (T, ensemble_n_clips, C, H, W)
            vid_frm_array = self._load_video_multi_clips_uniform(vid_id)
        else:
            vid_frm_array, _ = self._load_video(vid_id)  # tensor (T, C, H, W)
        batch["vid"] = vid_frm_array
        return batch

    def _prepare_batches_by_video(self):
        """create batches where each batch contains a single video with multiple text"""
        text_list = []
        for d in self.datalist:
            text_list.append(dict(
                text_str=d["txt"],
                id=d["id"],
            ))
        text_batch = dict(
            vid_id=None,
            examples=text_list,
            n_examples=len(text_list),
            ids=[d["id"] for d in text_list]
        )

        # make 1000 batches for 1000video x 1000text combinations.
        # each batch contains 1video x 1000text
        batches = []
        for d in self.datalist:
            _batch = copy.deepcopy(text_batch)
            _batch["vid_id"] = d["vid_id"]
            batches.append(_batch)
        return batches


class ClipBertMSRVTTMCEvalDataset(ClipBertBaseDataset):
    """ Each video is paired with 5 candidate captions, with only one correct.

    datalist: list(dict), each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=40, ensemble_n_clips=1):
        self.ensemble_n_clips = ensemble_n_clips
        super(ClipBertMSRVTTMCEvalDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.id2answer = {d["id"]: int(d["answer"]) for d in datalist}
        self.id2data = {d["id"]: d for d in self.datalist}

    def __len__(self):
        return len(self.datalist)

    def _load_video_multi_clips_uniform(self, vid_id):
        vid_frm_array_list = []
        video_max_pts = None
        for clip_idx in range(self.ensemble_n_clips):
            frames, video_max_pts = self._load_video(
                vid_id,
                num_clips=self.ensemble_n_clips,
                clip_idx=clip_idx, safeguard_duration=True,
                video_max_pts=video_max_pts)
            vid_frm_array_list.append(frames)
        return torch.cat(vid_frm_array_list, dim=0)

    def __getitem__(self, index):
        item = self.datalist[index]
        item["qid"] = item["id"]
        item["label"] = item["answer"]
        item["options_str_list"] = item["options"]
        del item["options"]
        # add videos  skip error videos:
        vid_id = item["vid_id"]
        if self.ensemble_n_clips > 1:
            # tensor (T*ensemble_n_clips, C, H, W), reshape as (T, ensemble_n_clips, C, H, W)
            vid_frm_array = self._load_video_multi_clips_uniform(vid_id)
        else:
            vid_frm_array, _ = self._load_video(vid_id)  # tensor (T, C, H, W)
        return dict(
            vid=vid_frm_array,
            examples=[item],
            n_examples=1  # used to create image feature copies.
        )

    def evaluate_qa_accuracy(self, pred_id2answer, force_same=True):
        """
        Args:
            pred_id2answer: dict, {id: pred_answer_idx}
            force_same: bool, if True, the predictions should contain the same set of ids as the GT.
        """
        gt_ids = list(self.id2answer.keys())
        pred_ids = list(pred_id2answer.keys())
        print(f"There are {len(gt_ids)} gt ids, {len(pred_ids)} pred ids.")
        if force_same:
            assert set(gt_ids) == set(pred_ids)
            shared_ids = list(set(gt_ids) & set(pred_ids))
        else:
            shared_ids = pred_ids

        gt_answers = np.array([self.id2answer[k] for k in shared_ids])
        pred_answers = np.array([pred_id2answer[k] for k in shared_ids])
        acc = np.mean(gt_answers == pred_answers)
        return dict(mc_accuracy=f"{100 * acc:.2f}")


class MSRVTTMCCollator(object):
    def __init__(self, tokenizer, max_length=20):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        text_str_list = flat_list_of_lists([d["options_str_list"] for d in text_examples])  # (B * n_options, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        question_ids = [d["qid"] for d in text_examples]
        collated_batch = dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,  # list(int), example ids,
            meta=dict(text_examples=text_examples, text_str_list=text_str_list),
            labels=None,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
        return collated_batch


