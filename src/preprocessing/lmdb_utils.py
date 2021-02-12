import lmdb
import cv2
import os
import multiprocessing as mp
from tqdm import tqdm
from os.path import join


def read_raw_binary_file(file_path):
    """can actually be any file"""
    with open(file_path, 'rb') as f:
        return f.read()


def read_raw_binary_from_pair(id_path_pair):
    file_id, filepath = id_path_pair
    try:
        file_byte_str = read_raw_binary_file(filepath)
        return file_id, file_byte_str
    except Exception as e:
        return filepath, None


def get_abspaths_by_ext(dir_path, ext=(".jpg",)):
    """Get absolute paths to files in dir_path with extensions specified by ext.
    Note this function does work recursively.
    """
    if not isinstance(ext, tuple):
        ext = tuple([ext, ])
    filepaths = [os.path.join(root, name)
                 for root, dirs, files in os.walk(dir_path)
                 for name in files
                 if name.endswith(tuple(ext))]
    return filepaths


def read_compress_raw_img(img_path):
    """
    Args:
        img_path: str, path to a raw image (.jpg, ....)
    """
    img = cv2.imread(img_path)  # np array, uint8
    encoded_img_arr = cv2.imencode('.jpg', img)[1]
    return encoded_img_arr


def read_raw_img_from_pair(id_path_pair):
    img_id, img_path = id_path_pair
    try:
        encoded_img_arr = read_compress_raw_img(img_path)
        return img_id, encoded_img_arr
    except Exception as e:
        return img_path, None


def write_lmdb_from_id_path(
        id_path_pairs, lmdb_save_dir, num_workers,
        lmdb_preprocessing_fn=read_raw_img_from_pair):
    """
    Write
    Args:
        id_path_pairs: list(tuple), each tuple is (file_id, filepath)
        lmdb_save_dir: str, path to save lmdb
        num_workers: int
        lmdb_preprocessing_fn: a function that takes two args [file_id (str), filepath (str)]
            and returns [file_id (byte string), value (byte string)] that will be used as
            key and value for lmdb txn.put(key=file_id, value=value).
            Additionally, this function should handle exceptions, where it should return key as
            the filepath, and return value as None.

    """
    env = lmdb.open(lmdb_save_dir, map_size=1024**4)
    txn = env.begin(write=True)
    error_filepaths = []
    if num_workers > 1:
        with mp.Pool(num_workers) as pool, tqdm(total=len(id_path_pairs)) as pbar:
            for idx, (key, value) in enumerate(
                    pool.imap_unordered(
                        lmdb_preprocessing_fn, id_path_pairs, chunksize=128)):
                if value is None:
                    error_filepaths.append(key)
                    continue
                txn.put(key=str(key).encode("utf-8"), value=value)
                if idx % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                pbar.update(1)
    else:
        for idx, pair in tqdm(enumerate(id_path_pairs), total=len(id_path_pairs)):
            key, value = lmdb_preprocessing_fn(pair)
            if value is None:
                error_filepaths.append(key)
                continue
            txn.put(key=str(key).encode("utf-8"), value=value)
            if idx % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)

    if len(error_filepaths) > 0:
        with open(join(lmdb_save_dir, "error_filepaths.log"), "w") as f:
            f.write("\n".join(error_filepaths))
        print(f"There are {len(error_filepaths)} files raised exceptions, "
              f"3 examples are {error_filepaths[:3]}")
    txn.commit()
    env.close()
