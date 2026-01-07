---
title: CryoFM2 Playground
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Demo for cryoFM post-processing application
---

### How-to use

```bash
conda create -n cryofm python=3.10 -y
conda activate cryofm
pip install -r requirements.txt
python app.py
```

**Note:** If you need to access Hugging Face through a proxy, make sure to exclude local addresses (localhost, 127.0.0.1) from the proxy settings. Example:

```bash
NO_PROXY="localhost,127.0.0.1,::1,.example.com,*.example.com" \
no_proxy="localhost,127.0.0.1,::1,.example.com,*.example.com" \
HTTP_PROXY="http://proxy.example.com:8080" \
HTTPS_PROXY="http://proxy.example.com:8080" \
http_proxy="http://proxy.example.com:8080" \
https_proxy="http://proxy.example.com:8080" \
python app.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
