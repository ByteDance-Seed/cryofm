# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gzip
import shutil
import subprocess
import os.path as osp
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import mrcfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from cryofm.data_sources.emdb import download_file
from cryofm.core.utils.shape_utils import get_pad_width

BASE_URL = "https://files.pdbj.org/pub/emdb/structures/"


def parse_args():
    parser = ArgumentParser(
        description="Prepare CryoFM1 dataset by downloading and preprocessing EMDB density maps.",
        epilog="""
Example usage:
  python scripts/prepare_cryofm1_dataset.py \\
      --data-list-path ./cryoFM-emdb-lists/cryofm1/cryofm1_1-5apix_dataset/test.csv \\
      --save-dir ./cryofm1_1-5apix_dataset
  
  The data-list-path should be a CSV file with an 'id' column containing EMDB IDs.
  CSV files are available on Zenodo (see project documentation for download links).
  The script will download maps, extract them, and rescale to 1.5 Angstrom/pixel.
        """
    )
    parser.add_argument(
        "--data-list-path", 
        type=str, 
        required=True,
        help="Path to CSV file containing EMDB IDs. The CSV must have an 'id' column with EMDB entry IDs (e.g., EMD-12345)."
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        required=True,
        help="Directory where downloaded and processed maps will be saved. The script will create subdirectories for each EMDB entry."
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=8,
        help="Number of parallel download workers for concurrent file downloads (default: 8). Increase for faster downloads if bandwidth allows."
    )
    return parser.parse_args()


def download_single_file(emdb_id, base_url, save_dir):
    """Download a single file for a given emdb_id."""
    file_name = emdb_id.lower().replace("-", "_") + ".map.gz"
    remote_path = osp.join(base_url, emdb_id, "map", file_name)
    cur_save_dir = osp.join(save_dir, str(emdb_id), "map", file_name)
    
    try:
        success = download_file(remote_path, cur_save_dir, cli="aria2c")
        return emdb_id, success, None
    except Exception as e:
        return emdb_id, False, str(e)


def gz_extract(src, dst):
    with gzip.open(src, 'rb') as f_in:
        with open(dst, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def _all_same(shape):
    return len(set(shape)) == 1


def pad_to_cubic(arr: np.ndarray):
    dst_shape = (max(arr.shape), ) * arr.ndim
    pad_width = get_pad_width(arr.shape, dst_shape)
    return np.pad(arr, pad_width, 'constant')


def preprocess(emdb_id, save_dir):
    map_f_name = f"{emdb_id.lower().replace('-', '_')}"
    map_rescale_name = map_f_name + "_rescale"
    cur_save_dir = osp.join(save_dir, str(emdb_id), "map")
    os.makedirs(cur_save_dir, exist_ok=True)

    gz_extract(osp.join(cur_save_dir, map_f_name + ".map.gz"), osp.join(cur_save_dir, map_f_name + ".map"))
    os.remove(osp.join(cur_save_dir, map_f_name + ".map.gz"))

    with mrcfile.open(osp.join(cur_save_dir, map_f_name + ".map")) as m:
        map_data = m.data
        voxel_size = m.voxel_size

    if not _all_same(map_data.shape):
        print(f"{emdb_id} shape {map_data.shape} pad to cubic")
        map_data = pad_to_cubic(map_data)

    with mrcfile.new(osp.join(cur_save_dir, map_f_name + ".mrc"), overwrite=True) as m:
        m.set_data(np.asarray(map_data, dtype=np.float32))
        m.voxel_size = voxel_size

    os.remove(osp.join(cur_save_dir, map_f_name + ".map"))

    new_voxel_size = 1.5
    subprocess.run(["relion_image_handler", "--rescale_angpix", f"{new_voxel_size}",
                    "--i", osp.join(cur_save_dir, map_f_name + ".mrc"),
                    "--o", osp.join(cur_save_dir, map_rescale_name + ".mrc")])
    os.remove(osp.join(cur_save_dir, map_f_name + ".mrc"))


if __name__ == "__main__":
    args = parse_args()

    data_list_path = args.data_list_path
    save_dir = args.save_dir

    os.makedirs(osp.join(save_dir, "split"), exist_ok=True)
    shutil.copyfile(data_list_path, osp.join(save_dir, "split", osp.basename(data_list_path)))
    
    # 1. Download deposited maps
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(data_list_path)
    id_list = df["id"].to_list()

    # Multi-threaded download
    num_workers = args.num_workers
    failed_downloads = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_id = {
            executor.submit(download_single_file, emdb_id, BASE_URL, save_dir): emdb_id
            for emdb_id in id_list
        }

        with tqdm(total=len(id_list), desc="Downloading") as pbar:
            for future in as_completed(future_to_id):
                emdb_id, success, error = future.result()
                if success:
                    pbar.set_postfix_str(f"Downloaded: {emdb_id}")
                else:
                    failed_downloads.append((emdb_id, error))
                    pbar.set_postfix_str(f"Failed: {emdb_id}")
                pbar.update(1)

    if failed_downloads:
        print(f"\nFailed to download {len(failed_downloads)} files:")
        for emdb_id, error in failed_downloads:
            print(f"  {emdb_id}: {error}")
    
    # 2. Extract and rescale deposited maps to 1.5Apix
    for emdb_id in id_list:
        preprocess(emdb_id, save_dir)
