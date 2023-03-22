# TempSAL - Uncovering Temporal Information for Deep Saliency Prediction

![teaser-colord](https://user-images.githubusercontent.com/16324609/226619656-7aca1b74-0746-4524-9a5b-cd71698d30ce.png)
An example of how human attention evolves over time. Top row: Temporal (shown in orange) and image (shown in pink) saliency ground truth from the SALICON dataset. Bottom row: Our temporal and image saliency predictions. Each temporal saliency map $\mathcal{T}_i$, $i \in \{1,\ldots,5\}$ represents one second of observation time. Note that in $\mathcal{T}_1$, the chef is salient, while in  $\mathcal{T}_2$ and  $\mathcal{T}_3$, the food on the barbecue becomes the most salient region in this scene. We can predict the temporal saliency maps for each interval separately, or combine them to create a single, refined image saliency map for the entire observation period.  





[![arXiv](https://img.shields.io/badge/arXiv-2301.02315-b31b1b.svg)](: https://arxiv.org/abs/2301.02315)

Project page and Supplementary material : https://ivrl.github.io/Tempsal/

We will relase the code on this repository upon publication.

## Citation

If you make use of our work, please cite our paper:

```
@InProceedings{pajouheshgar2022dynca,
  title     = {TempSAL - Uncovering Temporal Information for Deep Saliency Prediction},
  author    = {Aydemir, Bahar and Hoffstetter, Ludo and Zhang, Tong and Salzmann, Mathieu and S{\"u}sstrunk, Sabine},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
}
```








# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
