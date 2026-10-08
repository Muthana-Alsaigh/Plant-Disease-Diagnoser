# PlantVillage Disease Classification (MLOps Pipeline)

## 1. Six-Field Problem Statement

| Field | Description |
| :--- | :--- |
| **Input x** | Plant leaf image (RGB image, rescaled to 224x224 pixels) with metadata. |
| **Output y** | Predicted class label for plant disease (e.g., Tomato___healthy, Tomato___Early_blight). |
| **Prediction time $t_0$** | Real-time inference upon API request (`/predict`) or batch evaluation time before retraining. |
| **Decision** | Automated alert/recommendation system advising farmers on early crop treatment actions based on severity. |
| **Metric** | Macro F1-Score, Classification Accuracy ($\ge 90\%$), and Precision/Recall per class. |
| **Constraints** | Low inference latency ($<200\text{ms}$), model binary size constraints ($<100\text{MB}$), CPU/GPU memory footprint during serving. |

---