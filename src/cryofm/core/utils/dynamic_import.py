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

import re
import sys
import importlib.util


# powered by GPT
def path_to_module_name(path: str) -> str:
    """
    Converts a file path into a valid Python module name by replacing any non-alphanumeric characters with underscores.

    @param path: The file path to be converted.
    @return: A valid Python module name as a string.
    """
    # Regular expression matching any non-alphanumeric character.
    non_alphanumeric_pattern = re.compile('[^a-zA-Z0-9]')

    # Replace all non-alphanumeric characters in the path with underscores
    module_name = non_alphanumeric_pattern.sub('_', path)

    # Make sure the module name does not start with a digit
    if module_name[0].isdigit():
        module_name = '_' + module_name

    return module_name


# powered by GPT
def import_module_from_path(path: str, add_to_sys: bool = True):
    """
    Dynamically imports a module from the given file path.

    @param path: The full file path of the module.
    @param add_to_sys: Add imported module to the sys.modules dictionary.
    @return: The imported module object.
    """
    # Generate a unique module name based on the file path to avoid conflicts.
    module_name = path_to_module_name(path)

    # Check if the module has been already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Load module from the specified path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Cannot load module from path '{path}'")
    module = importlib.util.module_from_spec(spec)

    # Optionally, add the module to sys.modules
    if add_to_sys:
        sys.modules[module_name] = module

    # Execute the module (run its code)
    spec.loader.exec_module(module)

    return module
