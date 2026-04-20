import torch
import clip
import shutil
import os
import numpy as np
import sys
import yaml
import glob

from collections import defaultdict
from skimage.metrics import structural_similarity

sys.path.append(os.path.abspath('.'))

import utils.eval_utils as eu

def get_project_path():
    if os.getcwd().endswith("scripts"):
        return("../")
    else:
        return("./")

def create_directory_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_video_path(results_path, video_name, date_path, positive_prompts):
    gif_list = []
    for i in range(len(positive_prompts)):
        general_path = f'{results_path}{date_path[i]}/{video_name}/'
        all_direct = os.listdir(general_path)
        direct = [p for p in all_direct if p.startswith(positive_prompts[i])][0]
        file_path = general_path + direct + "/"
        file = [f for f in os.listdir(file_path) if f.endswith(".gif") or f.endswith(".mp4")][0]
        gif_list.append(file_path + file)
    return gif_list

def save_new_scores(prompt, results_path, save_path, shuffling_type, no_prompt_evaluation=False, video_name=None, extension='gif'):
    if results_path == "./":
        name_gif = glob.glob(os.path.join(save_path, f"*.{extension}"))[0]
        results_dict = evaluate_results(name_gif, {shuffling_type: {}}, prompt, no_prompt_evaluation, video_name=video_name, extension=extension)
    else:
        results_dict = evaluate_results(results_path, {shuffling_type: {}}, prompt, no_prompt_evaluation, video_name=video_name, extension=extension)
    yaml_dict = {'shuffle': shuffling_type}
    for key in results_dict[shuffling_type].keys():
        try:
            yaml_dict[key] = float(str(results_dict[shuffling_type][key]))
        except:
            yaml_dict[key] = float(str(results_dict[shuffling_type][key].item()))
    results_file_name = f'{save_path}/results.yaml'
    if os.path.exists(results_file_name):
        with open(results_file_name, 'r') as old_res_file:
            old_scores = yaml.safe_load(old_res_file)
        old_scores = dict(old_scores)
        nb_old = 0
        for old_key in old_scores.keys():
            if old_key not in yaml_dict.keys():
                yaml_dict[old_key] = old_scores[old_key]
                nb_old += 1
        print(f"keep {nb_old} old metrics")
    with open(f'{save_path}/results.yaml', 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)

def save_scores(target_prompt, save_path, video_name, shuffling_type):
    extension = "gif"
    if target_prompt == "":
        prompt = "no_prompt"
        no_prompt_evaluation = True
    else:
        prompt = target_prompt
        no_prompt_evaluation = False
    split_path = save_path.split('/')
    prepare_result_directory([shuffling_type], video_name, prompt, [split_path[-3]],
                                    [split_path[-1]])
    results_path = f'./results/{video_name}/{prompt}/'
    save_new_scores(prompt, results_path, save_path, shuffling_type, no_prompt_evaluation, video_name=video_name, extension=extension)


def prepare_result_directory(methods_dict, video_name, prompt, date_path, positive_prompts, source=''):
    project_path = get_project_path()
    results_path = project_path + "results/"

    # find path of edited videos
    gif_list = find_video_path(results_path, video_name, date_path, positive_prompts)

    # group all these files in a directory
    general_directory = f'{results_path}/{video_name}/{prompt}/'
    create_directory_if_needed(f'{results_path}/{video_name}/')
    create_directory_if_needed(general_directory)
    # create files of the form results/video_name/prompt/method.gif
    for (i,video) in enumerate(gif_list):
        shutil.copy(video, f'{general_directory}{methods_dict[i]}.{video.split(".")[-1]}')
    # copy the original .mp4 video
    if source == '':
        mp4_path = project_path + "data/" + video_name + ".mp4"
    else:
        mp4_path = project_path + "data/" + source + ".mp4"

    shutil.copy(mp4_path, f'{general_directory}source.mp4')


def evaluate_results(results_path, methods_dict, prompt, no_prompt_evaluation=False, video_name=None, extension='gif'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    raft_model = eu.prepare_raft_model(device)

    for k in methods_dict.keys():
        scores = defaultdict(float)
        if results_path.endswith('.gif'):
            video_path = results_path
        else:
            video_path = f'{results_path}/{k}.{extension}'
        source_video_path = f"./data/{video_name}.mp4"
        try:
            pil_list = eu.video_to_pil_list(video_path)
            if pil_list == []:
                pil_list = eu.video_to_pil_list(project_path + video_path)
        except:
            pil_list = eu.video_to_pil_list(project_path + video_path)
        source_pil_list = eu.video_to_pil_list(source_video_path)
        if source_pil_list == []:
            source_pil_list = eu.video_to_pil_list(project_path + source_video_path)

        if no_prompt_evaluation:
            scores['clip_similarity'] = eu.clip_similarity(pil_list, source_pil_list, preprocess, device, model)
            scores['lpips_similarity'] = eu.lpips_similarity(pil_list, source_pil_list, device)
            scores['warp-error-ssim'] = eu.warp_video(pil_list, source_pil_list, raft_model, device, structural_similarity)
        else:
            scores['clip-frame'] = eu.clip_frame(pil_list, preprocess, device, model)
            scores['clip-text'] = eu.clip_text(pil_list, prompt, preprocess, device, model)
            scores['ssim'] = eu.ssim(pil_list, device, structural_similarity)
            scores['warp-error-ssim'] = eu.warp_video(pil_list, source_pil_list, raft_model, device, structural_similarity)

        methods_dict[k] = scores.copy()

    for k in methods_dict.keys():
        print(f'\t{k}: ', end='')
        for s in sorted(methods_dict[k].keys()):
            print(f'{methods_dict[k][s]:.4f}', end=', ')
        print()
    print()

    return methods_dict


if __name__ == '__main__':
    project_path = get_project_path()

    evaluation_videos = ["bear"]
    results_path = f"{project_path}results/04-17-2026/"

    for video in evaluation_videos:
        if video in os.listdir(results_path):
            for dir in os.listdir(f"{results_path}{video}/"):
                print(f"Evaluation of {results_path}{video}/{dir} in progress...")
                single_res = f"{results_path}{video}/{dir}" # without "/" at the end
                prompt = dir.split("-")[0]
                shuffling_type = "mallowsAdaptive"
                if os.path.exists(f"{single_res}/results.yaml"):
                    with open(f'{single_res}/results.yaml', 'r') as yaml_file:
                            shuffling_type = dict(scores)["shuffle"]
                save_scores(prompt, single_res, video, shuffling_type)