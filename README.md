# 🦅 Bird vs Drone Classifier

> 🧠 A deep learning model built with Fastai to distinguish between **birds** and **drones** with 98.79% validation accuracy.

This project is designed to help in drone detection, airspace monitoring, and wildlife protection by leveraging a custom image classification model trained on aerial imagery.

---

## 🚀 Key Features

- ✅ Built with **Fastai** and **ResNet34**
- 🎯 Achieved **98.79% accuracy** on the validation set
- 📁 Dataset from [Kaggle](https://www.kaggle.com/datasets/harshwalia/birds-vs-drone-dataset)
- 🔍 Real-time prediction-ready with `.pkl` export

---

## 📂 Project Files

| File                     | Description                             |
|--------------------------|-----------------------------------------|
| `bird_vs_drone_model.pkl`| Trained model ready for inference       |
| `bird_vs_drone.ipynb`    | Full training + evaluation notebook     |
| `README.md`              | This file                               |
| `LICENSE`                | MIT License                             |
| `requirements.txt`       | Dependencies list                       |

---

## 🧪 Results Snapshot

| Epoch | Accuracy  | Validation Loss |
|-------|-----------|------------------|
| 3     | **98.79%** | 0.0760          |

---

## 🖥️ Inference Example

```python
from fastai.vision.all import *

learn = load_learner('bird_vs_drone_model.pkl')
pred, pred_idx, probs = learn.predict('example.jpg')
print(f"Prediction: {pred} — Confidence: {probs[pred_idx]:.2%}")
```

---

## 📦 Requirements

```bash
fastai>=2.7
torch
pillow
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📖 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software with proper attribution.

---

## 🙌 Credits

- [Fastai](https://www.fast.ai/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/harshwalia/birds-vs-drone-dataset)
