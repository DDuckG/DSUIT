import os
import torch
from typing import Tuple, Dict, Any
from models.bisenet_v2.lib.models.bisenetv2 import BiSeNetV2

def _strip_module_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    return { (k[len("module."): ] if k.startswith("module.") else k) : v for k,v in sd.items() }

def _is_flat_state_dict(sd: Any) -> bool:
    if not isinstance(sd, dict) or len(sd) == 0:
        return False
    sample = next(iter(sd.values()))
    return hasattr(sample, 'shape') or torch.is_tensor(sample)

def load_bisenet_from_vendor(weights_path: str = None, device: str = "cuda", num_classes: int = 19, verbose: bool = True) -> Tuple[BiSeNetV2, dict]:
    weights_path = os.path.join(os.path.dirname(__file__), "weights", "bisenetv2_cityscapes.pth")
    device_t = torch.device(device if torch.cuda.is_available()  else "cpu")

    try:
        model = BiSeNetV2(num_classes, aux_mode='eval')
    except TypeError:
        model = BiSeNetV2(num_classes)

    ckpt = torch.load(weights_path, map_location='cpu')

    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model' in ckpt):
        sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint content is not a dict/state_dict.")

    sd = _strip_module_prefix(sd)

    info = {
        'ckpt_total_keys': len(sd),
        'matched_keys': 0,
        'matched_key_names': [],
        'unmatched_ckpt_keys_sample': [],
        'unmatched_model_keys_sample': []
    }

    sample_val = next(iter(sd.values()))
    if isinstance(sample_val, dict):
        model_children = dict(model.named_children())
        matched = []
        unmatched_ckpt = []
        for child_name, child_state in sd.items():
            if child_name in model_children and isinstance(child_state, dict):
                try:
                    model_children[child_name].load_state_dict(child_state, strict=True)
                    matched.append(child_name)
                except Exception:
                    try:
                        model_children[child_name].load_state_dict(child_state, strict=False)
                        matched.append(child_name)
                    except Exception:
                        unmatched_ckpt.append(child_name)
            else:
                unmatched_ckpt.append(child_name)
        info['matched_keys'] = len(matched)
        info['matched_key_names'] = matched[:200]
        info['unmatched_ckpt_keys_sample'] = unmatched_ckpt[:200]
    else:
        model_sd = model.state_dict()
        matched = []
        unmatched_ckpt = []
        for k_ck, v_ck in sd.items():
            if k_ck in model_sd and model_sd[k_ck].shape == v_ck.shape:
                model_sd[k_ck] = v_ck
                matched.append(k_ck)
            else:
                unmatched_ckpt.append(k_ck)
        model.load_state_dict(model_sd, strict=False)
        info['matched_keys'] = len(matched)
        info['matched_key_names'] = matched[:200]
        info['unmatched_ckpt_keys_sample'] = unmatched_ckpt[:200]
        info['unmatched_model_keys_sample'] = [k for k in model_sd.keys() if k not in matched][:200]

    model.to(device_t)
    model.eval()

    if verbose:
        print("[BISENET LOADER] checkpoint keys:", info['ckpt_total_keys'])
        print("[BISENET LOADER] matched_keys:", info['matched_keys'])
        if info['matched_keys'] < info['ckpt_total_keys']:
            print("[BISENET LOADER] sample unmatched ckpt keys:", info['unmatched_ckpt_keys_sample'][:30])
            print("[BISENET LOADER] sample unmatched model keys:", info.get('unmatched_model_keys_sample', [])[:30])

    return model, info
