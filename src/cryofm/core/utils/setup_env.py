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

def register_custom_modules() -> None:
    """
    Registers custom modules (datasets, transforms, etc.) into global registries so they become 
    available for configuration-based instantiation throughout the CryoFM codebase.

    Typically used before model or trainer instantiation, to ensure all dynamic plugins are 
    registered and ready for use.

    Returns
    -------
    None

    Examples
    --------
    >>> from cryofm.core.utils.setup_env import register_custom_modules
    >>> register_custom_modules()
    >>> # Now custom datasets/transforms can be built from config
    """
    import cryofm.core.datasets  # noqa: F401,F403
    import cryofm.core.datasets.transforms  # noqa: F401,F403
