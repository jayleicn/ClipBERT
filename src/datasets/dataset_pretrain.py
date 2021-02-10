import torch
import random
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.data_utils import mask_batch_text_tokens
from src.datasets.dataset_base import ClipBertBaseDataset, img_collate


class ClipBertPretrainDataset(ClipBertBaseDataset):
    """
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict {
            "type": "image",
            "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
            "text": "A plate of food and a beverage are on a table.",  # should be tokenized and digitized first?
            ...
            }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    vis_format: str, image or video, used to decide data loading method.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, itm_neg_prob=0.5,
                 use_itm=True, vis_format="image"):
        super(ClipBertPretrainDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm, frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.itm_neg_prob = itm_neg_prob
        self.use_itm = use_itm
        self.vis_format = vis_format

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        if self.vis_format == "image":
            # one image/video with multiple examples
            vis_id, examples = self.datalist[index]
            img_array = self._load_img(vis_id)  # tensor, (T=1, C, H, W)
        else:  # video
            num_retries = 3  # skip error videos
            for _ in range(num_retries):
                vis_id, examples = self.datalist[index]
                img_array, _ = self._load_video(vis_id)  # tensor, (T=num_frm, C, H, W)
                # Select a random video if the current video was not able to access.
                if img_array is None:
                    LOGGER.info(f"Failed to load examples with video: {vis_id}. "
                                f"Will randomly sample an example as a replacement.")
                    index = random.randint(0, len(self) - 1)
                    continue
                else:
                    break
            else:
                raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        examples = [self._get_single_example(e, index) for e in examples]
        return dict(
            img=img_array,  # (T, C, H, W)
            examples=examples,
            n_examples=len(examples)  # used to create image feature copies.
        )

    def _get_single_example(self, data, index):
        # sample itm
        # random.random is uniform distributed in [0.0, 1.0)
        if self.use_itm and random.random() < self.itm_neg_prob:
            text_str = self._get_random_negative_caption(index)
            itm_label = 0  # negative pair
        else:
            text_str = data["txt"]
            itm_label = 1  # positive pair
        return dict(
            text_str=text_str,
            itm_label=itm_label
        )

    def _get_random_negative_caption(self, gt_index):
        gt_img_id, _ = self.datalist[gt_index]
        max_trials = 5
        while max_trials > 0:
            neg_index = int(random.random() * len(self))
            neg_img_id, neg_examples = self.datalist[neg_index]
            if neg_img_id == gt_img_id:
                max_trials -= 1
                continue
            else:
                break
        if max_trials == 0:
            LOGGER.info(f"gt_filepath {gt_img_id} index {gt_index}, "
                        f"neg_data filepath {neg_examples} index {neg_index}")
            raise Warning(f"The negative sampler cannot sample a true negative within 5 trials")
        neg_data = neg_examples[int(random.random() * len(neg_examples))]
        return neg_data["txt"]


class PretrainCollator(object):
    """is_train is kept here if we want to remove
    the randomness during validation of MLM accuracy.
    In that case, instantiate two PretrainCollator"""
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15,
                 max_length=20, is_train=True):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.is_train = is_train

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
        if self.mlm:
            text_input_ids, mlm_labels = mask_batch_text_tokens(
                text_input_ids, self.tokenizer,
                is_train=self.is_train)  # make mlm data
        else:
            text_input_ids, mlm_labels = text_input_ids, None
        text_input_mask = batch_enc.attention_mask  # (B, L)
        itm_labels = default_collate(
            [d["itm_label"] for d in text_examples])  # (B, )
        return dict(
            visual_inputs=visual_inputs,  # (B, #frm=1 or T, H, W, C)
            text_input_ids=text_input_ids,
            mlm_labels=mlm_labels,
            text_input_mask=text_input_mask,
            itm_labels=itm_labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
