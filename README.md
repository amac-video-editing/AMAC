# AMAC: Adaptive Multi-frame sAmpling for Consistent zero-shot video editing

## Abstract
Ensuring a convincing temporal coherence is one of the main challenges in zero-shot video editing. To address this issue, this paper presents a new method called AMAC, that can achieve both temporal consistency and detail preservation. The method relies on an adaptive sampling strategy to select frames that are jointly processed using a pre-trained text-to-image diffusion model. By reformulating the sampling strategy as a stochastic permutation over the frame indexes and constructing its distribution according to inter-frame similarities, we promote the consistent processing of related content together. This method demonstrates better robustness against changes and shot transitions, and is particularly well suited to edit long dynamic video sequences, as shown through experiments on the DAVIS and BDD100K datasets.

## AMAC overview
[![PDF Thumbnail](fig/AMAC-overview.png)](fig/AMAC-overview.pdf)

## AMAC results
The `generated_videos` directory provides generated videos of our method AMAC against baselines across different datasets. The directory is structured as follows:

- **`generated_videos/DAVIS/`**  
  - Contains GIF files where each file displays **the source video** alongside **the edited video produced by our method AMAC**.
  - Illustrates Section 5.2 **Short term editing** of the paper.

- **`generated_videos/BDD100K-UserStudy/`**  
  - Contains MP4 files sourced from our **User Study**, with added method name. 
  - Each MP4 presents **the source video**, followed by **the edited videos from two baseline methods and our method**.  
  - The **title of each file** specifies the order of the videos inside.
  - Illustrates Section 5.2 **Dynamic video editing** of the paper.

- **`generated_videos/Toy example/`**  
  - Contains MP4 files showcasing **the source video**, **the edited videos from two baselines**, and **our method's result**.  
  - The **order of the videos** within each file is explicitly described in the **file titles**: source video is at the top, followed by VidToMe, then RAVE, and finally AMAC at the bottom.
  - Illustrates Section 5.3 **Robustness to abrupt changes** of the paper.

Here are shape and style editing on 36-frame and 90-frame DAVIS videos:

<table>
<tr>
  <td>
    <figure>
      <img src="generated_videos/DAVIS/a pitbull_source-AMAC.gif">
      <figcaption>36-frame shape editing</figcaption>
    </figure>
  </td>
  <td>
    <figure>
      <img src="generated_videos/DAVIS/a dog is moving on a snowy field_source-AMAC.gif">
      <figcaption>36-frame style editing</figcaption>
    </figure>
  </td>
  <td>
    <figure>
      <img src="generated_videos/DAVIS/a black cat is running_source-AMAC.gif">
      <figcaption>90-frame shape editing</figcaption>
    </figure>
  </td>
</tr>
</table>
