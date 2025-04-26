# 🦵 Knee Osteoarthritis Detection using Deep Learning

This project implements a deep learning pipeline to detect **knee osteoarthritis severity** from X-ray images. It uses a fine-tuned **ResNet-18** model to classify knee conditions into three severity levels:  
- **0**: Normal  
- **1**: Mild/Moderate  
- **2**: Severe  

Relabeling is based on the original dataset grades (0–4), which are merged into three classes for better generalization.

---

## 📁 Dataset Structure

Make sure your dataset is organized like this:

```
dataset/
├── train/
│   ├── 0/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
│   ├── 0/
│   ├── 2/
│   ├── 3/
│   └── 4/
└── test/
    ├── 0/
    ├── 2/
    ├── 3/
    └── 4/
```

Relabel mapping:
```
0 → 0  
1, 2 → 1  
3, 4 → 2
```

---

## 🚀 Features

- PyTorch-based training pipeline  
- Data augmentation and preprocessing  
- Weighted loss for imbalanced classes  
- Model checkpointing  
- Evaluation using accuracy and F1-score  
- Supports GPU acceleration  

---

## 🧠 Model

- **Base Model**: ResNet-18 (pretrained on ImageNet)  
- **Modified**: Final FC layer changed to match 3 output classes  
- **Loss Function**: Cross Entropy with class weights  
- **Optimizer**: Adam  
- **Scheduler**: ReduceLROnPlateau  

---

## 🧪 Evaluation

Metrics used for evaluation:
- **Accuracy**
- **F1-score** (per class)

Model is evaluated on a separate test set and saved using `pickle` for deployment or future inference.

---

## 🛠 How to Run

1. Update the dataset path in the code:
   ```python
   path = "A:/project_x/Flask-Knee-Osteoarthritis-Classification/dataset/"
   ```
2. Run the training script:
   ```bash
   python train.py
   ```
3. Evaluate the model on the test set:
   ```bash
   python test.py
   ```

---

## 📦 Dependencies

- `torch`, `torchvision`, `PIL`  
- `sklearn`, `pandas`, `numpy`  
- `matplotlib`, `seaborn`  
- `tqdm`, `cv2`

Install with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- You can integrate this with a Flask app for deployment.  
- Make sure to load `model_pickle.pkl` during inference.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

