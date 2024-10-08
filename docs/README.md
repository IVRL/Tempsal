# TempSAL - Uncovering Temporal Information for Deep Saliency Prediction - CVPR 2023
🔥 Check out the demo here: [Hugging Face Space - TempSAL ](https://huggingface.co/spaces/baharay/tempsal)
![teaser-colord](https://user-images.githubusercontent.com/16324609/226619656-7aca1b74-0746-4524-9a5b-cd71698d30ce.png)
An example of how human attention evolves over time. Top row: Temporal (shown in orange) and image (shown in pink) saliency ground truth from the SALICON dataset. Bottom row: Our temporal and image saliency predictions. Each temporal saliency map $\mathcal{T}_i$, $i \in \{1,\ldots,5\}$ represents one second of observation time. Note that in $\mathcal{T}_1$, the chef is salient, while in  $\mathcal{T}_2$ and  $\mathcal{T}_3$, the food on the barbecue becomes the most salient region in this scene. We can predict the temporal saliency maps for each interval separately, or combine them to create a single, refined image saliency map for the entire observation period.  

![til](https://github.com/IVRL/Tempsal/blob/1bcfecb7d15fe284b5125c929a31ca6465b5247a/docs/rowa%20(1).gif)



Paper: (https://openaccess.thecvf.com/content/CVPR2023/papers/Aydemir_TempSAL_-_Uncovering_Temporal_Information_for_Deep_Saliency_Prediction_CVPR_2023_paper.pdf)

Project page and Supplementary material: https://ivrl.github.io/Tempsal/

# Installing required packages

Install the packages with pip using the following command under src/ folder.

```pip install -r requirements.txt```

# Inference
Download the model checkpoint from:
https://drive.google.com/drive/folders/1W92oXYra_OPYkR1W56D80iDexWIR7f7Z?usp=sharing 

Follow the instructions on inference.ipynb. This notebook provides predictions on temporal and image saliency together.

# Data
Download temporal saliency ground-truth saliency maps and fixations produced from the SALICON dataset :
https://drive.google.com/drive/folders/1afangzz2JFxRfRkQ-shjnhp8OyJCXL3G?usp=drive_link

Alternatively, you can use generate_volumes.py to produce temporal saliency slices in desired intervals&numbers.

# Temporal saliency only
For temporal saliency training and predictions, see:
https://github.com/LudoHoff/TemporalSaliencyPrediction

## Citation

If you make use of our work, please cite our paper:

```
@InProceedings{aydemir2023tempsal,
  title     = {TempSAL - Uncovering Temporal Information for Deep Saliency Prediction},
  author    = {Aydemir, Bahar and Hoffstetter, Ludo and Zhang, Tong and Salzmann, Mathieu and S{\"u}sstrunk, Sabine},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
}
```








# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
