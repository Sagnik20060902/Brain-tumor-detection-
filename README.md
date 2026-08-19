#  Brain Tumor Detection using Deep Learning (PyTorch)


An artificial intelligence computer vision pipeline designed to automate the classification of brain tumors from medical MRI scans. Built using PyTorch, this project leverages a custom ResNet deep convolutional network topology to aid in early automated radiological screenings.

---

##  Project Overview

Accurate analysis of neuro-imaging plays a key role in oncology diagnosis workflows. This project establishes an end-to-end medical deep learning pipeline inside a structured notebook environment to preprocess scans, train deep layers, and extract classification features.

### Key Features
- **ResNet Architecture Implementation:** Utilizes deep residual learning convolutional blocks (`BasicBlock`) to handle degradation problems during training.
- **Complete Iteration Tracking:** Monitors training vs. validation loss and accuracy scores sequentially through every epoch.
- **Robust Model Diagnostics:** Computes clear validation checks on unseen data splits to evaluate operational generalized capacity.
- **Cloud Optimized Pipeline:** Designed to run quickly and natively inside Kaggle or Google Colab environments using GPU runtime acceleration.

---

## Repository Structure

```text
Brain-tumor-detection-/
│
├── tumorecrdetection.ipynb   # Main PyTorch deep learning pipeline
└── README.md                 # Project documentation (this file)
```

---

## Tech Stack & Requirements

This system runs on Python 3.x and relies on standard deep learning/data science dependencies:

- **Framework:** `PyTorch` (Torchvision)
- **Mathematical Utilities:** `NumPy` & `Pandas`
- **Data Visualization:** `Matplotlib` (Pyplot)
- **Computer Vision Helper:**  `PIL`

---

## Setup & Usage Instruction

### 1. Replicate Local Repository
```bash
git clone https://github.com
cd Brain-tumor-detection-
```

### 2. Configure Environment & Python Dependencies
Ensure your local `pip` environment or conda engine is updated, then install:
```bash
pip install torch torchvision numpy pandas matplotlib opencv-python notebook
```

### 3. Execution Run
Open the server console to execute cells:
```bash
jupyter notebook
```
Launch `tumorecrdetection.ipynb` and execute the blocks sequentially.

---

## Model Architecture Details
The custom network implementation utilizes a residual neural mapping backbone built out of stacked `BasicBlock` modules containing:
- Dual structural 2D Convolution layers (`nn.Conv2d`) with matching padding configurations.
- Integrated Batch Normalization layers (`nn.BatchNorm2d`) for stabilization.
- Non-linear rectified linear unit activations (`nn.ReLU`).
- Downsampling layer bypass connections to ensure identity mappings are preserved safely across deeper dimensions.

---

## Performance Summary

The model was evaluated over 10 complete training epochs with tracking checkpoints:

| Phase Metric | Achievement Value |
| :--- | :--- |
| **Peak Training Accuracy** | 97.81% |
| **Final Test Dataset Accuracy** | **87.43%** |
| **Final Evaluated Test Loss** | 9.0642 |

*Note: The model shows signs of variance (overfitting) between training and evaluation phases. Future versions will incorporate additional dropout configurations, explicit weight decays, and stronger pixel-level augmentations to shrink this performance gap.*

---

## Contributing

Any optimization proposals or fixes are welcome! 
1. Fork this Repository.
2. Form your feature branch (`git checkout -b feature/Optimization`).
3. Commit modified changes (`git commit -m 'Add regularization fixes'`).
4. Push your updates (`git push origin feature/Optimization`).
5. Open an official Pull Request.

