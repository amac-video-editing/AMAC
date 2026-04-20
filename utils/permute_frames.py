import math
import torch
import random
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from torch.nn.functional import cosine_similarity

import utils.feature_utils as fu


def compute_clusters(list_frames, shuffling_type, total_frame_number, grid_frame_number):
    frames_array = np.stack([frame.view(-1).numpy() for frame in list_frames])
    param = shuffling_type.split("kmeans")[-1]
    if param == "Adaptive":
        max_k = max(math.ceil(total_frame_number / (grid_frame_number * 2)), 10)
        silhouettes = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(frames_array)
            silhouettes.append(silhouette_score(frames_array, labels))
        # Silhouette method: max score
        silhouette_k = range(2, max_k + 1)[np.argmax(silhouettes)]
        k = silhouette_k
        print("Number of clusters:", k)
    else:
        k = int(param)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(frames_array)
    return kmeans.labels_

def compute_similarities(list_frames):
    similarities = []
    for i in range(len(list_frames)):
        similarity = [cosine_similarity(list_frames[i], previous_frames, dim=0).item() for previous_frames in list_frames[:i + 1]]
        similarities.append(similarity)
    return similarities


def kmeans_permutation(nb_frames, clusters):
    indices = [0]
    for i in range(1, nb_frames):
        same_cluster = [k for k in range(i + 1) if clusters[k] == clusters[i]]
        random_draw = random.choice(same_cluster)
        indices.insert(random_draw, i)
    return indices

def mallows_permutation(nb_frames, similarities, scale, param=None):
    indices = [0]
    if param == "Adaptive":
        max_value = max([max(sub_lst) for sub_lst in similarities])
        threshold = np.mean([value for sublist in similarities for value in sublist])
    for i in range(1, nb_frames):
        values = list(range(i, -1, -1))
        if param == "Adaptive":
            values = list(range(i + 1))
            similarities_i = [val / max_value for val in similarities[i]]
            num = [val if val >= threshold else 0 for val in similarities_i]
        else:
            num = [math.exp(-j * scale) for j in range(i + 1)]
        probabilities = [ele / sum(num) for ele in num]
        random_draw = random.choices(values, probabilities)[0]
        indices.insert(random_draw, i)
    return indices


def plot_heatmap_matrix(heatmap_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    # exponential colormap
    exponent = 0.00001
    base_rgb = np.array(plt.cm.get_cmap("Reds")(np.linspace(0, 1, 256)))[:, :3]  # Extract RGB
    weights = np.linspace(0, 1, 256) ** exponent  # Exponential weighting
    colors = np.column_stack([(1 - weights) + weights * base_rgb[:, i] for i in range(3)])  # Interpolation
    exp_cmap = ListedColormap(colors)
    # graduations on legend
    cbar_kws = dict(ticks=[.0, heatmap_matrix.max() / 2, heatmap_matrix.max()])
    ax = sns.heatmap(heatmap_matrix, cmap=exp_cmap, annot=False, cbar_kws=cbar_kws, ax=ax)
    # axes graduations
    tick_positions = np.arange(0, 400, 100)  # Generate ticks at every 100
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(tick) for tick in tick_positions])
    ax.set_yticklabels([str(tick) for tick in tick_positions])
    plt.title(f"Heatmap of permutation matrices")
    plt.xlabel("Indices")
    plt.ylabel("Permuted indices")
    # Save with transparent background (optional)
    plt.savefig(
        f"./results/{datetime.datetime.now().strftime('%m-%d-%Y')}/adaptive_mallows_heatmap_permutation-matrix.png")
    plt.clf()


def shuffle_latents(latents, control_image, indices, nb_frames, sample_size, grid_frame_number, grid_size, type='mallowsAdaptive', clusters=None, similarities=None, no_prompt_latents=None):

    if type == 'random':
        ordered_i = torch.randperm(nb_frames).tolist()

    elif type.startswith('kmeans'):
        ordered_i = kmeans_permutation(nb_frames, clusters)

    elif type.startswith('mallows'):
        q = type.split('mallows')[1]
        if q == "Adaptive":
            ordered_i = mallows_permutation(nb_frames, similarities, -1, param=q)
            # plot heatmap permutation matrix
            n = len(ordered_i)
            num_samples = 100
            heatmap_matrix = np.zeros((n, n))
            # Generate permutation matrices and accumulate
            for i in range(num_samples):
                perm = mallows_permutation(nb_frames, similarities, -1, param=q)
                perm_matrix = np.zeros((n, n))
                perm_matrix[np.arange(n), perm] = 1
                heatmap_matrix += perm_matrix
            # Normalize by the number of samples
            heatmap_matrix /= num_samples
            plot_heatmap_matrix(heatmap_matrix)
        else:
            ordered_i = mallows_permutation(nb_frames, similarities, 1 - float(q))

    latents_l, controls_l, orderx = [], [], []
    if no_prompt_latents != None:
        no_prompt_latents_l = []
    for j in range(sample_size):
        my_indices = ordered_i[j * grid_frame_number:(j + 1) * grid_frame_number]
        latents_keyframe, _ = fu.prepare_key_grid_latents(latents, grid_size, grid_size, my_indices)
        control_keyframe, _ = fu.prepare_key_grid_latents(control_image, grid_size, grid_size, my_indices)
        latents_l.append(latents_keyframe)
        controls_l.append(control_keyframe)
        orderx.extend(my_indices)
        if no_prompt_latents != None:
            no_prompt_latents_keyframe, _ = fu.prepare_key_grid_latents(no_prompt_latents, grid_size, grid_size, my_indices)
            no_prompt_latents_l.append(no_prompt_latents_keyframe)

    ordered_i = orderx.copy()
    latents = torch.cat(latents_l, dim=0)
    control_image = torch.cat(controls_l, dim=0)
    indices = [indices[i] for i in ordered_i]
    if no_prompt_latents != None:
        no_prompt_latents = torch.cat(no_prompt_latents_l, dim=0)
        return latents, indices, control_image, no_prompt_latents

    return latents, indices, control_image