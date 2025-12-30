from __future__ import annotations

import os


_FALSEY = {"", "0", "false", "False", "no", "NO"}


def hf_is_offline() -> bool:
    """
    Returns True if the process should avoid network calls to Hugging Face Hub / transformers.

    We respect the common environment variables used by HF tooling:
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
      - HF_DATASETS_OFFLINE=1
    """
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        v = os.environ.get(key)
        if v is None:
            continue
        if str(v).strip() not in _FALSEY:
            return True
    return False


def hf_local_files_only() -> bool:
    """
    Convenience wrapper for `from_pretrained(..., local_files_only=...)`.
    """
    return hf_is_offline()

