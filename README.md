# Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration (AAAI2026)

[![Paper](https://img.shields.io/badge/arXiv-2504.09377-b31b1b.svg)](https://arxiv.org/abs/2504.09377) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of the paper "[Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration](https://arxiv.org/abs/2504.09377)".

📅 2025-12-22 New Experimental Results Added
* We have updated the experimental results by following the training and evaluation protocols of AdaIR and PromptIR for fair and comprehensive comparison.


## 📑 Abstract

All-in-one image restoration (AIR) aims to address diverse degradations within a unified model by leveraging informative degradation conditions to guide the restoration process. However, existing methods often rely on implicitly learned priors, which may entangle feature representations and hinder performance in complex or unseen scenarios. Histogram of Oriented Gradients (HOG) as a classical gradient representation, we observe that it has strong discriminative capability across diverse degradations, making it a powerful and interpretable prior for AIR. Based on this insight, we propose HOGformer, a Transformer-based model that integrates learnable HOG features for degradation-aware restoration. The core of HOGformer is a Dynamic HOG-aware Self-Attention (DHOGSA) mechanism, which adaptively models long-range spatial dependencies conditioned on degradation-specific cues encoded by HOG descriptors. To further adapt the heterogeneity of degradations in AIR, we propose a Dynamic Interaction Feed-Forward (DIFF) module that facilitates channel–spatial interactions, enabling robust feature transformation under diverse degradations. Besides, we propose a HOG loss to explicitly enhance structural fidelity and edge sharpness. Extensive experiments on a variety of benchmarks, including adverse weather and natural degradations, demonstrate that HOGformer achieves state-of-the-art performance and generalizes well to complex real-world scenarios.

## 📈 Results

<details>
<summary><b>📊 Setting I Results</b></summary>

![Setting I Results](https://github.com/user-attachments/assets/9421fb07-8fd0-465f-89bd-b5599a184fe2)

</details>

<details>
<summary><b>📊 Setting II Results</b></summary>

![Setting II Results](https://github.com/user-attachments/assets/c14b50c6-d10d-423b-8344-fc9b1fb283fb)

</details>

<details>
<summary><b>📊 Setting III Results</b></summary>

![Setting III Results](https://github.com/user-attachments/assets/e861f0b8-6d99-4d77-978e-7d180a7bc0ab)

</details>

<details>
<summary><b>📊 Setting IV Results</b></summary>

![Setting IV Results](https://github.com/user-attachments/assets/c5e23ab3-70d5-4a0e-bf77-7dc4ccbf3397)

</details>

## 🏗️ Framework

<details>
<summary><b>💡 Motivation</b></summary>

<div align="center">
  <p><b>Motivation</b></p>
  <img src="https://github.com/user-attachments/assets/eee2809c-8c4c-40b3-afbc-2c03317c71bc" alt="Motivation" width="85%">
</div>

</details>

<details>
<summary><b>🧠 Method Overview</b></summary>

<div align="center">
  <p><b>Method Overview</b></p>
  <img src="https://github.com/user-attachments/assets/88eea38b-5af5-4c00-abde-7325abbe808c" alt="Method" width="85%">
</div>

</details>

## 🛠️ Setup

### Installation

#### Setting I
```bash
# Clone the repository
git clone https://github.com/Fire-friend/HOGformer.git
cd HOGformer/settingI

# Create and activate virtual environment (recommended)
conda create -n HOGformerI python==3.10
conda activate HOGformerI

# Install dependencies
pip install -r requirements.txt

# Install basicsr
python setup.py develop --no_cuda_ext
````

## 📊 Dataset

### Data Preparation

#### Setting I

| Resource                            | Download Links                                                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Training dataset** (All together) | [BaiduYun Disk](https://pan.baidu.com/s/1LagvtxjK8BEJdJvl6ntSmg) (Code: m695) <br> [Google Drive (TBD)]() |
| **Test dataset** (All together)     | [BaiduYun Disk](https://pan.baidu.com/s/1ZZgOxKkVXBImtBWXOBg0LQ) (Code: nabu) <br> [Google Drive (TBD)]() |

#### Setting III & Setting IV 

Please follow the [AdaIR code](https://github.com/c-yn/AdaIR) for configuration and training.

## 🚀 Usage

### Training

#### Setting I

```bash
cd settingI
./train.sh Allweather/Options/Allweather_HOGformer.yml 4321
```

#### Setting III & Setting IV

Please follow the [AdaIR code](https://github.com/c-yn/AdaIR) for configuration and training.

### Evaluation

#### Setting I

1. Download the pretrained models:

   * [BaiduYun Disk](https://pan.baidu.com/s/17c-1eSklHNA6NmEznUjwug) (Code: wa6u)
   * [Google Drive (TBD)]()
2. Place the downloaded models in `./Allweather/pretrained_models/`
3. Test with the replaced argument:

   ```bash
   cd Allweather
   python test_histoformer.py --input_dir [INPUT_FOLDER] --result_dir result/ --weights pretrained_models/net_g_latest.pth --yaml_file Options/Allweather_HOGformer.yml
   ```

#### Setting III & Setting IV
Three Tasks:
* [BaiduYun Disk](https://pan.baidu.com/s/1dAlXezIDPejCDnab1u_xOw?pwd=w9yk)
* [Google Drive](https://drive.google.com/file/d/1QgmTPPZBtaRg6QFIK2OAM0QMXRtpKnSF/view?usp=sharing)

Five Tasks:
* [BaiduYun Disk](https://pan.baidu.com/s/1qoFIJVbE0oBc9wX91VFWtw?pwd=8as2)
* [Google Drive](https://drive.google.com/file/d/1KwSXF-seYqFrbaFh3ZQojCQD-DJ9sNyq/view?usp=sharing)

```bash
python test.py
```

## 📷 Visualizations

<details>
<summary><b>🔍 Visualization 1</b></summary>

![Visualization 1](https://github.com/user-attachments/assets/7bbd3a2d-6a88-4a7a-b1b8-ab7d9197541a)

</details>

<details>
<summary><b>🔍 Visualization 2</b></summary>

![Visualization 2](https://github.com/user-attachments/assets/3268651b-0581-4c92-b4db-0b8fe6038745)

</details>

## 📝 Citation

If you use our code or method in your research, please cite our paper:

```bibtex
@article{wu2025beyond,
  title={Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers},
  author={Wu, Jiawei and Yang, Zhifei and Wang, Zhe and Jin, Zhi},
  journal={arXiv preprint arXiv:2504.09377},
  year={2025}
}
@article{lastname2023paper,
  title={Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration},
  author={Wu, Jiawei and Yang, Zhifei and Wang, Zhe and Jin, Zhi},
  journal={AAAI},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions, please contact: [wujw97@mail2.sysy.edu.cn](mailto:wujw97@mail2.sysy.edu.cn)

---

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox, [Histoformer](https://github.com/sunshangquan/Histoformer), [DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR), and [AdaIR](https://github.com/c-yn/AdaIR).

