# AMAC: Adaptive Multi-frame sAmpling for Consistent zero-shot video editing

## Abstract
Ensuring a convincing temporal coherence is one of the main challenges in zero-shot video editing. To address this issue, this paper presents a new method called AMAC, that can achieve both temporal consistency and detail preservation. The method relies on an adaptive sampling strategy to select frames that are jointly processed using a pre-trained text-to-image diffusion model. By reformulating the sampling strategy as a stochastic permutation over the frame indexes and constructing its distribution according to inter-frame similarities, we promote the consistent processing of related content together. This method demonstrates better robustness against changes and shot transitions, and is particularly well suited to edit long dynamic video sequences, as shown through experiments on the DAVIS and BDD100K datasets.

## AMAC overview
[![PDF Thumbnail](fig/AMAC-overview.png)](fig/AMAC-overview.pdf)

## AMAC results
The `generated_videos` directory provides a clear comparison of our method against baselines across different datasets. The directory is structured as follows:

- **`generated_videos/DAVIS/`**  
  - Contains MP4 files where each file displays **the source video** alongside **the edited video produced by our method AMAC**.

- **`generated_videos/BDD100K-UserStudy/`**  
  - Contains GIF files sourced from our **user study**.  
  - Each GIF presents **the source video**, followed by **the edited videos from two baseline methods and our method**.  
  - The **title of each file** specifies the order of the videos inside.

- **`generated_videos/Toy example/`**  
  - Contains GIF files showcasing **the source video**, **the edited videos from two baselines**, and **our method's result**.  
  - The **order of the videos** within each GIF is explicitly described in the **file titles**: source video is in the top-left, VidToMe is in the top-right, RAVE is in the bottom-left, and AMAC is in the bottom-right.

Here are shape and style editing on 36-frame and 90-frame DAVIS videos.
<table>
<tr>
  <td><img src="generated_videos/DAVIS/a pitbull_source-amac.gif"></td>
  <td><img src="generated_videos/DAVIS/a dog is moving on a snowy field_source-amac.gif"></td>
  <td><img src="generated_videos/DAVIS/a black cat is running_source-amac.gif"></td>
</tr>
</table>
You can find more videos in `generated_videos` directory.
