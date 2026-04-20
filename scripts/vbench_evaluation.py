import os
import sys
import cv2
import yaml
import glob
import shutil
import json
import subprocess
import statistics
import numpy as np
from PIL import Image
from datetime import datetime


def create_directory_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        success, frame = video.read()
    video.release()
    return frames

def evaluate_and_save(path_dir, metric):
    edited_vid = glob.glob(os.path.join(path_dir, '*.gif')) + glob.glob(os.path.join(path_dir, '*.mp4'))

    subprocess.run(["vbench", "evaluate", "--dimension", metric, "--videos_path", edited_vid[0], "--mode=custom_input"])
    start_file_name = f"results_{datetime.now().strftime('%Y-%m-%d')}-"
    end_file_name = "_eval_results.json"
    time_lst = [file.split(start_file_name)[1].split(end_file_name)[0] for file in os.listdir("./evaluation_results/")
                if file.startswith(start_file_name) and file.endswith(end_file_name)]
    cur_file_name = start_file_name + max(time_lst) + end_file_name

    with open("./evaluation_results/" + cur_file_name, 'r') as res_file:
        res_dict = json.load(res_file)
    try:
        new_score = res_dict[metric][0]
    except:
        new_score = sum(rewards) / len(rewards)

    try:
        # load previous scores
        with open(path_dir + "/results.yaml", 'r') as res_file:
            scores = yaml.safe_load(res_file)
        # save new score
        scores[metric] = new_score
        with open(path_dir + "/results.yaml", 'w') as res_file:
            yaml.safe_dump(scores, res_file)
    except:
        print("go to next...")


if __name__ == '__main__':

    try:
        results_dir = sys.argv[1]
    except: # to run directly or debug the script
        results_dir = ("./results/04-17-2026/")

    shuffling_types = ['mallowsAdaptive']
    metrics = ["subject_consistency"]

    for video in os.listdir(results_dir):
        videos_path = f"{results_dir}{video}/"
        if os.path.isdir(videos_path):
            for sub_dir in os.listdir(videos_path):
                if os.path.isdir(videos_path + sub_dir):
                    for met in metrics:
                        print(f"{met} evaluation on {videos_path + sub_dir} in progress...")
                        evaluate_and_save(videos_path + sub_dir, met)
