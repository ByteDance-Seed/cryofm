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

# access info by xml, api, api(by key) will get different results, take care
import os
import gzip
import shutil
import tarfile
import requests
import subprocess
import os.path as osp
from typing import Union
from abc import ABC, abstractmethod

import xmltodict
import pandas as pd

EMDB_RSYNC_URL = "rsync.rcsb.org"


def load_xml_as_dict(file_path):
    with open(file_path) as xml_file:
        return xmltodict.parse(xml_file.read())


def dig_nested(nested, keys):
    """
    Search for given keys within a nested dictionary and/or list structure.

    If a key match is found within the current dictionary or list, the function will continue to
    search within the matched item for the next key.
    If a key match can't be found within the current dictionary or list, the function returns None.
    If a key is an index of a list, it will return the item at the corresponding position in the list.
    If a key is a function, it will return the first item that makes the function true.
    It goes through the list when the key is None. When 'nested' is a list, the function will return
    the first matched result all the time.

    Args:
        nested: The data structure (dictionary or list) to be searched.
        keys: List of keys to be searched for.

    Returns:
        Resulted item from depth search, if not found returns None.
    """
    # When no more keys left to search or if the nested structure
    # isn't a dictionary or list, end the search.
    if not keys or not isinstance(nested, (dict, list, tuple)):
        return nested

    key = keys[0]
    keys = keys[1:]

    # Normal dictionary search
    if isinstance(nested, dict) and key in nested:
        return dig_nested(nested[key], keys)
    elif isinstance(nested, (list, tuple)):
        # If the key is an index into the list
        if isinstance(key, int) and key < len(nested):
            return dig_nested(nested[key], keys)
        # If the key is a function
        elif callable(key):
            for item in nested:
                if key(item):
                    return dig_nested(item, keys)
        # In cases where the key is None, go through the list
        elif key is None:
            for item in nested:
                val = dig_nested(item, keys)
                if val is not None:
                    return val
    print(f"Key '{key}' not found.")
    return None


def download_file(url, save_path, cli="aria2c"):
    if cli == "aria2c":
        if url.endswith(".gz") or url.endswith(".map"):
            cmd = [cli, "-d", osp.dirname(save_path), "-q", "-s", "60", url]
        else:
            cmd = [cli, "-d", osp.dirname(save_path), "-q", url]
    elif cli == "wget":
        cmd = [cli, "-o", save_path, "-q", url]
    else:
        raise NotImplementedError(f"{cli}")
    try:
        subprocess.call(cmd)
        # print(f"File downloaded successfully in {save_path}.")

    except subprocess.CalledProcessError as error:
        print(f"An error occurred while downloading the file. {error}")


def extract_file(file_path, target_path=None):
    """
    Extract the specified file to the target directory.

    Supported formats:
    - .gz
    - .tar.gz
    - .tar

    Parameters:
    - file_path (str): Path to the file to be extracted.
    - target_path (str, optional): Destination directory for the extracted files. Defaults to the current working directory.
    """
    if target_path is None:
        target_path = os.getcwd()

    if not osp.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_lower = file_path.lower()

    try:
        if file_lower.endswith('.tar.gz') or file_lower.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=target_path)
            print(f"Extracted {file_path} to {target_path}")

        elif file_lower.endswith('.tar'):
            with tarfile.open(file_path, 'r:') as tar:
                tar.extractall(path=target_path)
            print(f"Extracted {file_path} to {target_path}")

        elif file_lower.endswith('.gz') and not (file_lower.endswith('.tar.gz') or file_lower.endswith('.tgz')):
            with gzip.open(file_path, 'rb') as f_in:
                output_file = osp.splitext(os.path.basename(file_path))[0]
                output_path = osp.join(target_path, output_file)
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted {file_path} to {output_path}")

        else:
            print(f"Unsupported file format: {file_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")


class EMDBRsyncClient:

    def __init__(self):
        self.base_url = "rsync.rcsb.org"

    def _run_cmd(self, cmd, shell=False):
        process = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        stdout = process.stdout
        return stdout

    def _list_dir(self, remote_dir_path: str, recursive=False):
        rsync_cmd = ['rsync', ]
        if recursive:
            rsync_cmd.append('-r')
        rsync_cmd.extend(["-lptz", "--list-only", "--port=33444", self.base_url + "::" + remote_dir_path])
        return self._run_cmd(rsync_cmd)

    def list_dir(self, remote_dir_path: str, recursive=False):
        ls_str = self._list_dir(remote_dir_path, recursive=recursive)
        df = parse_ls_str(ls_str)
        return df["File"].values.tolist()

    def list_only_files(self, remote_dir_path: str, recursive=False):
        ls_str = self._list_dir(remote_dir_path, recursive=recursive)
        df = parse_ls_str(ls_str)
        df = df[df["Permissions"].apply(lambda x: x[0] != 'd')]
        return df["File"].values.tolist()

    def sync(self, remote_path: str, local_path: str):
        self._run_cmd(['rsync', '-rlpt', '-v', '-z', '--delete', '--port=33444',
                       self.base_url + "::" + remote_path, local_path])


def parse_ls_str(ls_str: str):
    # Example line: drwxr-xr-x          61 2023/04/29 01:12:20 EMD-0001
    file_list = [line.split() for line in ls_str.split("\n") if line]
    df = pd.DataFrame(file_list, columns=["Permissions", "Size", "Date", "Time", "File"])
    return df


def get_valid_emdb_ids():
    file_names = EMDBRsyncClient().list_dir("emdb/structures/EMD-*")
    ids = [f.removeprefix("EMD-") for f in file_names]

    ids = sorted(ids, key=int)
    return ids


class EMDBRestApi:

    def __init__(self):
        self.valid_keys = ("admin", "publications", "map", "supplement", "sample", "vitrification",
                           "imaging", "image_acquisition", "processing", "experiment", "fitted",
                           "analysis", "annotations")

        self.base_url = "https://www.ebi.ac.uk/emdb/api"

    def _get(self, api):
        url = self.base_url + api
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_info(self, entry_id: str, key: str = None):
        if key is None:
            return self._get(f'/entry/{entry_id}')
        else:
            if key not in self.valid_keys:
                raise KeyError(f'Invalid key {key}.')
            return self._get(f'/entry/{key}/{entry_id}')


class _BaseInfo(ABC):
    properties = ["entry_id", "sample_name", "experiment_method", "shape", "apix", "contour_level",
                  "resolution", "num_half_maps", "pdb_ids", "protein_seqs"]

    @abstractmethod
    def _retrieve_info(self):
        ...

    def _parse_info_by_chain_keys(self, keys):
        tmp = self._retrieve_info()
        if tmp is not None:
            return dig_nested(tmp, keys)


class EMDBOnlineInfo(_BaseInfo):

    def __init__(self, entry_id: str):
        self.entry_id = entry_id
        self._info_cache = None
        self._info_keys_dict = {
            "sample_name": ["sample", "name", "valueOf_"],
            "experiment_method": ["structure_determination_list", "structure_determination", None, "method"],
            "shape": ["map", "spacing", "x"],
            "apix": ["map", "pixel_spacing", "x"],
            "contour_level": ["map", "contour_list", "contour", lambda x: x['primary'], "level"],
            "resolution": ["structure_determination_list", "structure_determination", None,
                           "image_processing", None, "final_reconstruction", "resolution"],
            "num_half_maps": ["interpretation", "half_map_list", "half_map"],
            "pdb_ids": ["crossreferences", "pdb_list", "pdb_reference"],
            "protein_seqs": ["sample", "macromolecule_list", "macromolecule"]
        }

    @staticmethod
    def _parse_unit_value(d, type_fn=None):
        # TODO: maybe there are other units like nm, ...
        if d["units"] != "\u212b":
            print(f"Unit is {d['units']} not A, skipped.")
            return None

        val = d["valueOf_"]
        if type_fn is not None:
            val = type_fn(val)
        return val

    def _retrieve_info(self):
        if self._info_cache is None:
            try:
                tmp = EMDBRestApi().get_info(self.entry_id)
            except Exception as e:
                print(e)
                return None
            else:
                self._info_cache = tmp

        return self._info_cache

    @property
    def sample_name(self) -> Union[str, None]:
        return self._parse_info_by_chain_keys(self._info_keys_dict["sample_name"])

    @property
    def experiment_method(self) -> Union[str, None]:
        return self._parse_info_by_chain_keys(self._info_keys_dict["experiment_method"])

    @property
    def shape(self) -> Union[int, None]:
        tmp = self._parse_info_by_chain_keys(self._info_keys_dict["shape"])
        if tmp is not None:
            tmp = int(tmp)
        return tmp

    @property
    def apix(self) -> Union[float, None]:
        apix = self._parse_info_by_chain_keys(self._info_keys_dict["apix"])

        if apix is not None:
            apix = self._parse_unit_value(apix, float)
        return apix

    @property
    def contour_level(self) -> Union[float, None]:
        contour_lvl = self._parse_info_by_chain_keys(self._info_keys_dict["contour_level"])
        if contour_lvl is not None:
            contour_lvl = float(contour_lvl)
        return contour_lvl

    @property
    def resolution(self) -> Union[float, None]:
        res = self._parse_info_by_chain_keys(self._info_keys_dict["resolution"])

        if res is not None:
            res = self._parse_unit_value(res, type_fn=float)
        return res

    @property
    def num_half_maps(self) -> Union[int, None]:
        half_maps = self._parse_info_by_chain_keys(self._info_keys_dict["num_half_maps"])

        num_half_maps = None
        if half_maps is not None:
            num_half_maps = len(half_maps)
        return num_half_maps

    @property
    def pdb_ids(self) -> list:
        pdb_ids = self._parse_info_by_chain_keys(self._info_keys_dict["pdb_ids"])

        if pdb_ids is not None:
            pdb_ids = [item['pdb_id'] for item in pdb_ids]
        else:
            pdb_ids = []
        return pdb_ids

    @property
    def protein_seqs(self) -> list:
        seq_list = self._parse_info_by_chain_keys(self._info_keys_dict["protein_seqs"])

        prot_seq_list = []
        if seq_list is not None:
            # some proteins use external_references in InterPro, like EMD-1005, here ignore them
            prot_seq_list = [item["sequence"]["string"].replace(' ', '').replace('\n', '') for item in seq_list
                             if item["instance_type"] == "protein_or_peptide" and "string" in item["sequence"]]

        return prot_seq_list


class EMDBOfflineInfo(EMDBOnlineInfo):

    def __init__(self, entry_id: str, local_db_path: str = None):
        assert local_db_path is not None
        super().__init__(entry_id)
        self.local_db_path = local_db_path

        self._info_keys_dict["sample_name"] = ["sample", "name"]
        self._info_keys_dict["experiment_method"] = ["structure_determination_list", "structure_determination",
                                                     "method"]
        self._info_keys_dict["contour_level"] = ["map", "contour_list", "contour", "level"]
        self._info_keys_dict["resolution"] = ["structure_determination_list", "structure_determination",
                                              f"{self.experiment_method.lower()}_processing", "final_reconstruction",
                                              "resolution"]
        self._info_keys_dict["protein_seqs"] = ["sample", "macromolecule_list", "protein_or_peptide"]

    @staticmethod
    def _parse_unit_value(d, type_fn=None):
        # TODO: maybe there are other units like nm, ...
        if d["@units"] != "\u212b":
            print(f"Unit is {d['units']} not A, skipped.")
            return None

        val = d["#text"]
        if type_fn is not None:
            val = type_fn(val)
        return val

    def _retrieve_info(self):
        if self._info_cache is None:
            file_names = [f"emd-{self.entry_id}-v30.xml", f"emd-{self.entry_id}.xml"]

            file_path = next((osp.join(self.local_db_path, name) for name in file_names if
                              osp.exists(osp.join(self.local_db_path, name))), None)

            if file_path is None:
                print(f"{file_names} not exists.")
            else:
                tmp = load_xml_as_dict(file_path)['emd']
                self._info_cache = tmp

        return self._info_cache

    @property
    def pdb_ids(self) -> list:
        pdb_ids = self._parse_info_by_chain_keys(self._info_keys_dict["pdb_ids"])

        if pdb_ids is not None:
            if isinstance(pdb_ids, dict):
                pdb_ids = [pdb_ids['pdb_id'], ]
            else:
                pdb_ids = [item['pdb_id'] for item in pdb_ids]
        else:
            pdb_ids = []
        return pdb_ids

    @property
    def protein_seqs(self) -> list:
        seq_list = self._parse_info_by_chain_keys(self._info_keys_dict["protein_seqs"])

        prot_seq_list = []
        try:
            if seq_list is not None:
                if (isinstance(seq_list, dict) and seq_list["sequence"] is not None
                        and "string" in seq_list["sequence"]):
                    prot_seq_list = [seq_list["sequence"]["string"].replace('\n', ''), ]
                elif isinstance(seq_list, list):
                    prot_seq_list = [item["sequence"]["string"].replace('\n', '') for item in seq_list
                                     if item is not None and item["sequence"] is not None
                                     and "string" in item["sequence"]]
        except Exception as e:
            print(self.entry_id, e)

        return prot_seq_list


# deprecated
class EMDBOnlineInfoV1:
    properties = ["entry_id", "sample_name", "experiment_method", "shape", "apix", "contour_level",
                  "resolution", "num_half_maps", "pdb_ids", "protein_seqs"]

    def __init__(self, entry_id: str):
        self.entry_id = entry_id
        self._info_cache = {}
        self._api_error = {}

    def _retrieve_info(self, key):
        if key in self._info_cache:
            return self._info_cache[key]
        else:
            try:
                tmp = EMDBRestApi().get_info(self.entry_id, key)
            except Exception as e:
                print(e)
                self._api_error[key] = True
                return None
            else:
                self._info_cache[key] = tmp
                return tmp

    @property
    def sample_name(self) -> str:
        tmp = self._retrieve_info("sample")

        name = dig_nested(tmp, ["sample", "name", "valueOf_"])
        return name

    @property
    def experiment_method(self) -> str:
        tmp = self._retrieve_info("vitrification")

        method = dig_nested(tmp, ["structure_determination_list", "structure_determination", None, "method"])
        return method

    @property
    def shape(self) -> int:
        tmp = self._retrieve_info("map")

        shape = dig_nested(tmp, ["map", "spacing", 'x'])
        if shape is not None:
            shape = int(shape)
        return shape

    @property
    def apix(self) -> float:
        tmp = self._retrieve_info("map")

        apix = dig_nested(tmp, ["map", "pixel_spacing", 'x'])
        if apix is not None:
            apix = EMDBOnlineInfo._parse_unit_value(apix, type_fn=float)
        return apix

    @property
    def contour_level(self) -> float:
        tmp = self._retrieve_info("map")

        contour_lvl = dig_nested(tmp, ["map", "contour_list", "contour", lambda x: x['primary'], "level"])
        if contour_lvl is not None:
            contour_lvl = float(contour_lvl)
        return contour_lvl

    @property
    def resolution(self) -> float:
        tmp = self._retrieve_info("vitrification")

        res = dig_nested(tmp, ["structure_determination_list", "structure_determination", None,
                               "image_processing", None, "final_reconstruction", "resolution"])

        if res is not None:
            res = EMDBOnlineInfo._parse_unit_value(res, type_fn=float)
        return res

    @property
    def num_half_maps(self) -> int:
        tmp = self._retrieve_info("supplement")

        half_maps = dig_nested(tmp, ["interpretation", "half_map_list", "half_map"])

        num_half_maps = None
        if half_maps is not None:
            num_half_maps = len(half_maps)
        return num_half_maps

    @property
    def pdb_ids(self) -> list:
        tmp = self._retrieve_info("fitted")

        pdb_ids = dig_nested(tmp, ["crossreferences", "pdb_list", "pdb_reference"])
        if pdb_ids is not None:
            pdb_ids = [item['pdb_id'] for item in pdb_ids]
        return pdb_ids

    @property
    def protein_seqs(self) -> list:
        tmp = self._retrieve_info("sample")

        seq_list = dig_nested(tmp, ["macromolecule_list", "macromolecule"])

        prot_seq_list = []
        if seq_list is not None:
            for item in seq_list:
                if item["instance_type"] == "protein_or_peptide":
                    prot_seq_list.append(item["sequence"]["string"].replace(' ', ''))

        return prot_seq_list
