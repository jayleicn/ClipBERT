import torch
import random
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad as img_pad
from torchvision.transforms.functional import resize as img_resize
from torch.nn.functional import interpolate as img_tensor_resize
from torch.nn.functional import pad as img_tensor_pad
from torch.nn.modules.utils import _quadruple
from src.utils.basic_utils import flat_list_of_lists
import numbers
import numpy as np
from PIL import Image
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def mask_batch_text_tokens(
        inputs, tokenizer, mlm_probability=0.15, is_train=True):
    """ modified from transformers.data.data_collator
    Args:
        inputs: (B, L), 2D torch.Tensor, does not work for 1D. It has already been padded.
        tokenizer:
        mlm_probability: float
        is_train: if True use random masking, else mask tokens at fixed position to remove randomness in evaluation.
    """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(
            val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape,
        dtype=torch.long)  # len(tokenizer) == #vocab
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def image_to_tensor(image: np.ndarray, keepdim: bool = True) -> torch.Tensor:
    """Converts a numpy image to a PyTorch 4d tensor image.
    Args:
        image (numpy.ndarray): image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim (bool): If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`. Default: ``True``
    Returns:
        torch.Tensor: tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.
    """
    if not isinstance(image, (np.ndarray,)):
        raise TypeError("Input type must be a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape))

    return tensor.unsqueeze(0) if not keepdim else tensor


def get_padding(image, max_w, max_h, pad_all=False):
    # keep the images to upper-left corner
    if isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
    else:
        w, h = image.size
    h_padding, v_padding = max_w - w, max_h - h
    if pad_all:
        h_padding /= 2
        v_padding /= 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    else:
        l_pad, t_pad = 0, 0
        r_pad, b_pad = h_padding, v_padding
    if isinstance(image, torch.Tensor):
        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    else:
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class ImagePad(object):
    def __init__(self, max_w, max_h, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.max_w = max_w
        self.max_h = max_h
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        if isinstance(img, torch.Tensor):
            paddings = _quadruple(get_padding(img, self.max_w, self.max_h))
            return img_tensor_pad(
                img, paddings,
                self.padding_mode, self.fill)
        return img_pad(
            img, get_padding(img, self.max_w, self.max_h),
            self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def get_resize_size(image, max_size):
    """
    Args:
        image: PIL Image or torch.tensor
        max_size:

    Returns:

    Note the height/width order difference
    >>> pil_img = Image.open("raw_img_tensor.jpg")
    >>> pil_img.size
    (640, 480)  # (width, height)
    >>> np_img = np.array(pil_img)
    >>> np_img.shape
    (480, 640, 3)  # (height, width, 3)
    """
    # note the order of height and width for different inputs
    if isinstance(image, torch.Tensor):
        # width, height = image.shape[-2:]
        height, width = image.shape[-2:]
    else:
        width, height = image.size

    if height >= width:
        ratio = width*1./height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height*1./width
        new_width = max_size
        new_height = new_width * ratio
    size = (int(new_height), int(new_width))
    return size


class ImageResize(object):
    """Resize the input image (torch.tensor) to the given size.

    Args:
        max_size (int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, max_size, interpolation=Image.BILINEAR):
        assert isinstance(max_size, int)
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (torch.tensor): Image to be scaled.

        Returns:
            torch.tensor: Rescaled image.
        """
        if isinstance(img, torch.Tensor):
            assert isinstance(self.interpolation, str)
            return img_tensor_resize(
                img, size=get_resize_size(img, self.max_size),
                mode=self.interpolation, align_corners=False)
        return img_resize(
            img, get_resize_size(img, self.max_size), self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


def get_imagenet_transform(min_size=600, max_size=1000):
    """parameters from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    This simply crop the center square from the image
    """
    if min_size != 600:
        import warnings
        warnings.warn(f'Warning: min_size is not used in image transform, '
                      f'setting min_size will have no effect.')
    return transforms.Compose([
        ImageResize(max_size, Image.BILINEAR),  # longer side will be resized to 1000
        ImagePad(max_size, max_size),  # pad to 1000 * 1000
    ])


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).cuda().view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 1, 3, 1, 1)
        # assert max(std) <= 1 and min(std) >= 0\
        #     or max(mean) <= 1 and min(mean) >= 0,\
        #         "Please provide mean or std within range [0, 1]"

    def __call__(self, img):
        """
        Args:
            img: float image tensors, (B, N, 3, H, W)

        Returns:
            img: normalized float image tensors
        """
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


def chunk_list(examples, chunk_size=2, pad_to_divisible=True):
    """
    Args:
        examples: iterable, examples grouped by image/video
        chunk_size: int, number of examples in each chunk.
        pad_to_divisible: bool, pad the examples to be divisible by chunk_size.
    >>> test_examples = [3, 4, 5, 6, 7]
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=True)
    [[3, 4], [5, 6], [7, 7]]  # the lst element has some randomness
    >>> chunk_list(test_examples, chunk_size=2, pad_to_divisible=False)
    [[3, 4], [5, 6], [7]]
    """
    n_examples = len(examples)
    remainder = n_examples % chunk_size
    if pad_to_divisible and remainder > 0:
        n_pad = chunk_size - remainder
        pad = random.choices(examples, k=n_pad)  # with replacement
        examples = examples + pad
        n_examples = len(examples)
        remainder = 0
    chunked_examples = []
    n_chunks = int(n_examples / chunk_size)
    n_chunks = n_chunks + 1 if remainder > 0 else n_chunks
    for i in range(n_chunks):
        chunked_examples.append(examples[i*chunk_size: (i+1)*chunk_size])
    return chunked_examples


def mk_input_group(key_grouped_examples, max_n_example_per_group=2, is_train=True,
                   example_unique_key=None):
    """ Re-organize examples into groups. Each input group will have a single image paired
    with X (X=max_n_example_per_img) examples. Images with total #examples > X will be
    split into multiple groups. In the case a group has < X examples, we will copy
    the examples to make the group has X examples.
    Args:
        key_grouped_examples: dict, each key is image/video id,
            each value is a list(example) associated with this image/video
        max_n_example_per_group: int, pair max #examples with each image/video.
           Note that each image can have multiple groups.
        is_train: bool, if True, copy the examples to make sure each input
            group has max_n_example_per_group examples.
        example_unique_key: str, used to make sure no inputs are discarded by matching
            the input and output ids specified by `example_unique_key`
    """
    input_groups = []  # each element is (id, list(example))
    for k, examples in key_grouped_examples.items():
        chunked_examples = chunk_list(examples,
                                      chunk_size=max_n_example_per_group,
                                      pad_to_divisible=is_train)
        for c in chunked_examples:
            # if len(c) == 0:
            #     continue
            input_groups.append((k, c))

    if example_unique_key is not None:
        print(f"Using example_unique_key {example_unique_key} to check whether input and output ids m")
        # sanity check: make sure we did not discard any input example by accident.
        input_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e] for e in key_grouped_examples.values()])
        output_question_ids = flat_list_of_lists(
            [[sub_e[example_unique_key] for sub_e in e[1]] for e in input_groups])
        assert set(input_question_ids) == set(output_question_ids), "You are missing "
    return input_groups


def repeat_tensor_rows(raw_tensor, row_repeats):
    """ repeat raw_tensor[i] row_repeats[i] times.
    Args:
        raw_tensor: (B, *)
        row_repeats: list(int), len(row_repeats) == len(raw_tensor)
    """
    assert len(raw_tensor) == len(raw_tensor), "Has to be the same length"
    if sum(row_repeats) == len(row_repeats):
        return raw_tensor
    else:
        indices = torch.LongTensor(
            flat_list_of_lists([[i] * r for i, r in enumerate(row_repeats)])
        ).to(raw_tensor.device)
        return raw_tensor.index_select(0, indices)



