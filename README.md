# Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers

[![Paper](https://img.shields.io/badge/arXiv-2504.09377-red)](https://arxiv.org/abs/2504.09377) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of the paper "[Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers](https://arxiv.org/abs/2504.09377)" (Coming Soon).

## 📑 Abstract

All-in-one image restoration, which aims to address diverse degradations within a unified framework, is critical for practical applications. However, existing methods rely on predicting and integrating degradation conditions, which can misactivate degradation-specific features in complex scenarios, limiting their restoration performance. To address this issue, we propose a novel all-in-one image restoration framework guided by Histograms of Oriented Gradients (HOG), named HOGformer. By leveraging the degradation-discriminative capability of HOG descriptors, HOGformer employs a dynamic self-attention mechanism that adaptively attends to long-range spatial dependencies based on degradation-aware HOG cues. To enhance the degradation sensitivity of attention inputs, we design a HOG-guided local dynamic-range convolution module that captures long-range degradation similarities while maintaining awareness of global structural information. Furthermore, we propose a dynamic interaction feed-forward module, efficiently increasing the model capacity to adapt to different degradations through channel–spatial interactions. Extensive experiments across diverse benchmarks, including adverse weather and natural degradations, demonstrate that HOGformer achieves state-of-the-art performance and generalizes effectively to complex real-world degradations.

## 🏗️ Framework
<div align="center">
  <p><b>Motivation</b></p>
  <img src="https://github.com/user-attachments/assets/eee2809c-8c4c-40b3-afbc-2c03317c71bc" alt="Motivation" width="85%">
</div>
<div align="center">
  <p><b>Method Overview</b></p>
  <img src="https://github.com/user-attachments/assets/257fc0a2-fee3-4960-8d85-b5f45bf7ebda" alt="Method" width="85%">
</div>

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
```

#### Setting II
```bash
# Clone the repository
git clone https://github.com/Fire-friend/HOGformer.git
cd HOGformer/settingII

# Create and activate virtual environment (recommended)
conda create -n HOGformerII python==3.10
conda activate HOGformerII

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

### Data Preparation

#### Setting I

| Resource | Download Links |
|----------|---------------|
| **Training dataset** (All together) | [BaiduYun Disk](https://pan.baidu.com/s/1LagvtxjK8BEJdJvl6ntSmg) (Code: m695) <br> [Google Drive (TBD)]() |
| **Test dataset** (All together) | [BaiduYun Disk](https://pan.baidu.com/s/1ZZgOxKkVXBImtBWXOBg0LQ) (Code: nabu) <br> [Google Drive (TBD)]() |

#### Setting II (TBD)

| Resource | Download Links |
|----------|---------------|
| **Training dataset** | [BaiduYun Disk]() <br> [Google Drive]() |
| **Test dataset** | [BaiduYun Disk]() <br> [Google Drive]() |

## 🚀 Usage

### Training

#### Setting I
```bash
cd settingI
./train.sh Allweather/Options/Allweather_HOGformer.yml 4321
```

#### Setting II
```bash
python XXXX # TBD
```

### Evaluation

#### Setting I
1. Download the pretrained models: 
   - [BaiduYun Disk](https://pan.baidu.com/s/17c-1eSklHNA6NmEznUjwug) (Code: wa6u)
   - [Google Drive (TBD)]()
2. Place the downloaded models in `./Allweather/pretrained_models/`
3. Test with the replaced argument:
   ```bash
   cd Allweather
   python test_histoformer.py --input_dir [INPUT_FOLDER] --result_dir result/ --weights pretrained_models/net_g_latest.pth --yaml_file Options/Allweather_HOGformer.yml
   ```

#### Setting II
```bash
python XXXX # TBD
```

## 📈 Results

![Setting I Results](https://github.com/user-attachments/assets/a97973fb-3611-489c-9c25-a59098a96cb5)

## 📷 Visualizations

![Visualization 1](https://github.com/user-attachments/assets/7bbd3a2d-6a88-4a7a-b1b8-ab7d9197541a)

![Visualization 2](https://github.com/user-attachments/assets/3268651b-0581-4c92-b4db-0b8fe6038745)

## 📝 Citation

If you use our code or method in your research, please cite our paper:

```bibtex
@article{lastname2023paper,
  title={Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers},
  author={Wu, Jiawei and Yang, Zhifei and Wang, Zhe and Jin, Zhi},
  journal={arXiv preprint arXiv:2504.09377},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions, please contact: wujw97@mail2.sysy.edu.cn

---

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox, [Histoformer](https://github.com/sunshangquan/Histoformer), and [DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR).