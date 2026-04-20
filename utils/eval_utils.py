from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import torch.nn.functional as F
import cv2
import imageio
import torch
import clip
import warnings
import statistics
import numpy as np

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def video_to_pil_list(video_path):
    if video_path.endswith('.mp4'):
        vidcap = cv2.VideoCapture(video_path)
        pil_list = []
        while True:
            success, image = vidcap.read()
            if success:
                pil_list.append(Image.fromarray(image))
            else:
                break

        return pil_list
    elif video_path.endswith('.gif'):
        gif = imageio.get_reader(video_path)
        pil_list = []

        for frame in gif:
            pil_list.append(Image.fromarray(frame))

        return pil_list


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img,
                    sample_coords,
                    mode='bilinear',
                    padding_mode='zeros',
                    return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (
            y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp_rerender(feature,
              flow,
              mask=False,
              mode='bilinear',
              padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature,
                           grid,
                           mode=mode,
                           padding_mode=padding_mode,
                           return_mask=mask)


def clip_text(pil_list, text_prompt, preprocess, device, model):
    text = clip.tokenize([text_prompt]).to(device)

    images = []
    with torch.no_grad():
        text_features = model.encode_text(text)
        for pil in pil_list:
            image = preprocess(pil).unsqueeze(0).to(device)
            images.append(image)
        image_features = model.encode_image(torch.cat(images))
        scores = [torch.cosine_similarity(text_features, image_feature).item() for image_feature in image_features]

    score = sum(scores) / len(scores)
    
    return score

def clip_frame(pil_list, preprocess, device, model):
    images = []
    with torch.no_grad():
        for pil in pil_list:
            image = preprocess(pil).unsqueeze(0).to(device)
            images.append(image)
        
        image_features = model.encode_image(torch.cat(images))
        
    image_features = image_features.cpu().numpy()
    cosine_sim_matrix = cosine_similarity(image_features)
    neighbors_sim = np.triu(cosine_sim_matrix, k=1) - np.triu(cosine_sim_matrix, k=2)

    score = neighbors_sim.sum() / (len(pil_list) - 1)

    return score


def ssim(pil_list, device, distance_func):
    dist = []

    for i in range(len(pil_list) - 1):
        dist.append(distance_func(np.array(pil_list[i]), np.array(pil_list[i+1]), channel_axis=2, data_range=255))
    score = np.mean(np.array(dist))

    return score

def clip_similarity(pil_list, source_pil_list, preprocess, device, model):
    scores = []
    for i in range(len(pil_list)):
        edited_image = preprocess(pil_list[i]).unsqueeze(0).to(device)
        edited_feature = model.encode_image(edited_image)
        source_image = preprocess(source_pil_list[i]).unsqueeze(0).to(device)
        source_feature = model.encode_image(source_image)
        sub_score = torch.cosine_similarity(edited_feature, source_feature).item()
        scores.append(sub_score)

    score = sum(scores) / len(scores)

    return score

def lpips_similarity(pil_list, source_pil_list, device):
    pil_list = [image.resize(source_pil_list[0].size) for image in pil_list]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(device)
    transform = transforms.Compose([transforms.PILToTensor()])
    scores = []
    for i in range(len(pil_list)):
        edited_image = (transform(pil_list[i]) / 255).unsqueeze(0).to(device)
        source_image = (transform(source_pil_list[i]) / 255).unsqueeze(0).to(device)
        sub_score = lpips(edited_image, source_image)
        scores.append(sub_score)

    score = sum(scores) / len(scores)

    return score

def prepare_raft_model(device):
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model.eval()

    return model

def flow_warp(img: np.ndarray,
              flow: np.ndarray,
              filling_value: int = 0,
              interpolate_mode: str = 'nearest'):
    '''Use flow to warp img.

    Args:
        img (ndarray): Image to be warped.
        flow (ndarray): Optical Flow.
        filling_value (int): The missing pixels will be set with filling_value.
        interpolate_mode (str): bilinear -> Bilinear Interpolation;
                                nearest -> Nearest Neighbor.

    Returns:
        ndarray: Warped image with the same shape of img
    '''
    warnings.warn('This function is just for prototyping and cannot '
                  'guarantee the computational efficiency.')
    assert flow.ndim == 3, 'Flow must be in 3D arrays.'
    height = flow.shape[0]
    width = flow.shape[1]
    channels = img.shape[2]

    output = np.ones(
        (height, width, channels), dtype=img.dtype) * filling_value

    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = grid[:, :, 0] + flow[:, :, 1]
    dy = grid[:, :, 1] + flow[:, :, 0]
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

    if interpolate_mode == 'nearest':
        output[valid, :] = img[dx[valid].round().astype(int),
                               dy[valid].round().astype(int), :]
    elif interpolate_mode == 'bilinear':
        # dirty walkround for integer positions
        eps_ = 1e-6
        dx, dy = dx + eps_, dy + eps_
        left_top_ = img[np.floor(dx[valid]).astype(int),
                        np.floor(dy[valid]).astype(int), :] * (
                            np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                np.ceil(dy[valid]) - dy[valid])[:, None]
        left_down_ = img[np.ceil(dx[valid]).astype(int),
                         np.floor(dy[valid]).astype(int), :] * (
                             dx[valid] - np.floor(dx[valid]))[:, None] * (
                                 np.ceil(dy[valid]) - dy[valid])[:, None]
        right_top_ = img[np.floor(dx[valid]).astype(int),
                         np.ceil(dy[valid]).astype(int), :] * (
                             np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                 dy[valid] - np.floor(dy[valid]))[:, None]
        right_down_ = img[np.ceil(dx[valid]).astype(int),
                          np.ceil(dy[valid]).astype(int), :] * (
                              dx[valid] - np.floor(dx[valid]))[:, None] * (
                                  dy[valid] - np.floor(dy[valid]))[:, None]
        output[valid, :] = left_top_ + left_down_ + right_top_ + right_down_
    else:
        raise NotImplementedError(
            'We only support interpolation modes of nearest and bilinear, '
            f'but got {interpolate_mode}.')
    return output.astype(img.dtype)

def calculate_flow(pil_list, model, DEVICE):
    def load_image(imfile, DEVICE):
        img = np.array(imfile).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = (img / 255) * 2 - 1
        return img[None].to(DEVICE)

    flow_up_list = []
    with torch.no_grad():
        images = pil_list.copy()
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1, DEVICE)
            image2 = load_image(imfile2, DEVICE)
            flow_up = model(image1, image2)[-1]
            flow_up_list.append(flow_up.detach().squeeze().permute(1,2,0).cpu().numpy())
    return flow_up_list

def rerender_warp(img, flow, mode='bilinear'):
    expand = False
    if len(img.shape) == 2:
        expand = True
        img = np.expand_dims(img, 2)

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    dtype = img.dtype
    img = img.to(torch.float)
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
    res = flow_warp_rerender(img, flow, mode=mode)
    res = res.to(dtype)
    res = res[0].cpu().permute(1, 2, 0).numpy()
    if expand:
        res = res[:, :, 0]
    return res

def opencv_warp(img, flow):
    h, w = flow.shape[:2]
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    warped_img = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return warped_img

def warp_video(edit_pil_list, source_pil_list, raft_model, device, distance_func):
    source_pil_list = [image.resize(edit_pil_list[0].size) for image in source_pil_list]
    flow_up_list = calculate_flow(source_pil_list, raft_model, device)

    res_list = [edit_pil_list[0]]
    for i, pil_img in enumerate(edit_pil_list[:-1]):
        warped = opencv_warp(np.array(pil_img), flow_up_list[i])
        pil_warped = to_pil_image(warped)
        res_list.append(pil_warped)

    if distance_func == structural_similarity:
        scores = np.array([distance_func(np.array(edit_pil_list[i]), np.array(res_list[i]), channel_axis=2, data_range=255) for i in range(len(res_list))])
        return np.mean(scores)
    else:
        scores = np.array([distance_func(edit_pil_list[i], res_list[i]) for i in range(len(res_list))])
        return np.mean(scores)
