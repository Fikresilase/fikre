"""Shared helpers: seeding, IO, VRAM teardown."""
import gc
import json
import random

import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def free_vram(*objs):
    """Fully tear down model objects + KV cache between model passes."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    # vLLM-specific: drop the distributed/model-parallel state if present.
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel, destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
