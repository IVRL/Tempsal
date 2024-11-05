
# **🌟 TempSAL: Uncovering Temporal Information for Deep Saliency Prediction - CVPR 2023**

[![Demo on Hugging Face Space](https://img.shields.io/badge/Demo-Hugging%20Face%20Space-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/baharay/tempsal)

---

![Teaser Image](https://user-images.githubusercontent.com/16324609/226619656-7aca1b74-0746-4524-9a5b-cd71698d30ce.png)

> **Example of Evolving Human Attention Over Time:**  
> The top row shows temporal (in orange) and image (in pink) saliency ground truth from the SALICON dataset. The bottom row displays our predictions. Each temporal saliency map $\mathcal{T}_i$, where $i \in \{1,\ldots,5\}$, represents one second of observation time. Notably, in $\mathcal{T}_1$, the chef is the salient focus, while in $\mathcal{T}_2$ and $\mathcal{T}_3$, the food on the barbecue becomes the most salient region. Temporal saliency maps can be predicted for each interval individually or combined to produce a refined saliency map for the entire observation period.

![Temporal Saliency GIF](https://github.com/IVRL/Tempsal/blob/1bcfecb7d15fe284b5125c929a31ca6465b5247a/docs/rowa%20(1).gif)

---

## 📄 **Research Paper**
- [TempSAL - Uncovering Temporal Information for Deep Saliency Prediction - CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Aydemir_TempSAL_-_Uncovering_Temporal_Information_for_Deep_Saliency_Prediction_CVPR_2023_paper.pdf)

## 🌐 **Project Page and Supplementary Material**
- Visit the [TempSAL Project Page](https://ivrl.github.io/Tempsal/) for more resources and supplementary materials.

---

## 🚀 **Getting Started**

### 1. Installing Required Packages

Install all necessary packages by running the following command in the `src/` folder:
```bash
pip install -r requirements.txt
```

### 2. Inference

1. **Download Model Checkpoint:**  
   Download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1W92oXYra_OPYkR1W56D80iDexWIR7f7Z?usp=sharing).

2. **Run Inference:**  
   Follow instructions in `inference.ipynb` to generate predictions on both temporal and image saliency.

### 3. Data

- **Download Ground-Truth Data**  
  Temporal saliency ground-truth maps and fixation data from the SALICON dataset are available [here](https://drive.google.com/drive/folders/1afangzz2JFxRfRkQ-shjnhp8OyJCXL3G?usp=drive_link).
  
- **Generate Custom Saliency Volumes**  
  Alternatively, use `generate_volumes.py` to create temporal saliency slices with customizable intervals.

### 4. Temporal Saliency Only
For projects focused on temporal saliency training and predictions, please refer to [TemporalSaliencyPrediction](https://github.com/LudoHoff/TemporalSaliencyPrediction) by Ludo Hoff.

---

## 📜 **Citation**

If you use this work in your research, please cite our paper as follows:

```bibtex
@InProceedings{aydemir2023tempsal,
  title     = {TempSAL - Uncovering Temporal Information for Deep Saliency Prediction},
  author    = {Aydemir, Bahar and Hoffstetter, Ludo and Zhang, Tong and Salzmann, Mathieu and S{"u}sstrunk, Sabine},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
}
```

---

## 📜 **License**

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0; vertical-align:middle" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>  
This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

---
