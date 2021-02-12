import os
from src.preprocessing.lmdb_utils import write_lmdb_from_id_path, \
    read_raw_img_from_pair, read_raw_binary_from_pair


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


def get_filename(filepath):
    return os.path.splitext(os.path.split(filepath)[1])[0]


def main_convert():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        help="path to dir containing all files needed to convert")
    parser.add_argument("--lmdb_save_dir", type=str, help="path to dir saving the lmdb files")
    parser.add_argument("--num_workers", type=int, default=4, help="#workers / #threads")
    parser.add_argument("--ext", type=str, nargs="+", help="file ext to store in lmdb")
    parser.add_argument("--file_type", type=str, choices=["image", "video"], help="data type to store in lmdb")
    args = parser.parse_args()

    if os.path.exists(args.lmdb_save_dir) and os.listdir(args.lmdb_save_dir):
        raise ValueError(f"lmdb_save_dir {args.lmdb_save_dir} already exists and is not empty")
    else:
        os.makedirs(args.lmdb_save_dir, exist_ok=True)

    file_paths = get_abspaths_by_ext(dir_path=args.data_root, ext=args.ext)
    id_path_pairs = [(get_filename(p), p) for p in file_paths]
    lmdb_preprocessing_fn = read_raw_binary_from_pair \
        if args.file_type == "video" else read_raw_img_from_pair
    write_lmdb_from_id_path(
        id_path_pairs=id_path_pairs,
        lmdb_save_dir=args.lmdb_save_dir,
        lmdb_preprocessing_fn=lmdb_preprocessing_fn,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main_convert()
