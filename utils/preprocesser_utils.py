import cv2
import yaml
import numpy as np

from annotator.zoe import ZoeDetector
from annotator.raft import RaftDetector

def yaml_load(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def yaml_dump(path, data):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


def zoe_depth(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_zoe_depth = ZoeDetector()
    result = model_zoe_depth(img)
    return remove_pad(result), True

def raft(img, prev_image, res=512, **kwargs):
    model_raft = RaftDetector()
    result = model_raft(img, prev_image)
    return result, True

preprocessors_dict = {
    'depth_zoe': zoe_depth,
    'optical_flow': raft,
}

def pixel_perfect_process(input_image, p_name, prev_image=None):
    raw_H, raw_W, _ = input_image.shape
    preprocessor_resolution = raw_H
    if p_name == 'optical_flow':
        detected_map, _ = (preprocessors_dict[p_name](input_image, prev_image, res=preprocessor_resolution))
        detected_map = detected_map.cpu().numpy()
    else:
        detected_map, _ = preprocessors_dict[p_name](input_image, res=preprocessor_resolution)
    return detected_map
