# Early-Stage Parkinson's Disease Detection Using Keystroke Dynamics

**Data Science Project: Non-invasive monitoring of motor symptoms through passive typing behavior analysis with Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.x-FF6600?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blue)](https://github.com/slundberg/shap)

This data science research project implements a machine learning system that predicts early-stage Parkinson's Disease (PD) from natural, unconstrained keystroke dynamics collected in users' home environments. The project introduces a novel **Circadian HoldTime Fluctuation** feature that models the daily motor fluctuations characteristic of Parkinson's patients on Levodopa medication, and validates model predictions using **SHAP (SHapley Additive exPlanations)** to meet clinical transparency requirements.

---

## 🎯 Project Highlights

| **Key Achievement** | **Description** |
|---|---|
| **Novel Feature Engineering** | Created "Circadian HoldTime Fluctuation" to mathematically capture On/Off motor fluctuations by comparing morning vs. evening typing speeds |
| **Robust Ensemble Model** | Soft Voting Ensemble (Random Forest + XGBoost + SVM) achieved **69.2% accuracy** vs. 61.5% baseline on highly imbalanced data (75% healthy, 25% PD) |
| **Explainable AI Integration** | SHAP analysis validated that predictions are driven by clinically relevant physiological markers, not noise |
| **Unconstrained Data** | Analyzed **9+ million keystrokes** from 227 users typing naturally over several months — not controlled lab tasks |
| **Class Imbalance Handling** | Applied SMOTE (Synthetic Minority Over-sampling Technique) to address severe class imbalance |

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | F1-Score (Macro) |
|-------|----------|------------------|
| Logistic Regression (Baseline) | 61.5% | 55.8% |
| Random Forest (Tuned) | 61.5% | 43.6% |
| XGBoost | 61.5% | 51.1% |
| SVM (RBF) | 59.0% | 51.9% |
| **Soft Voting Ensemble** | **69.2%** | **59.7%** |

The **Soft Voting Ensemble** outperformed all individual classifiers by combining their probabilistic predictions, reducing algorithmic bias and providing continuous confidence scores rather than binary diagnostic labels.

### Visualization Gallery

| Confusion Matrix | ROC Curve |
|---|---|
| ![Confusion Matrix](assets/confusion_matrix.png) | ![ROC Curve](assets/roc_curve.png) |

| SHAP Summary Plot | Feature Importance |
|---|---|
| ![SHAP Summary](assets/shap_summary.png) | ![Feature Importance](assets/feature_importance.png) |

The SHAP analysis demonstrates that **Circadian HoldTime Fluctuation** is one of the strongest predictive features, confirming that the model learned clinically valid patterns rather than spurious correlations.

---

## 🧠 Why This Matters

**Traditional Parkinson's diagnosis relies on episodic clinical assessments (UPDRS) that:**
- Occur only during scheduled clinic visits
- Cannot capture day-to-day motor fluctuations ("On/Off" states)
- Are subjective and depend on the patient's state during the appointment

**This project demonstrates that:**
- Passive keystroke monitoring can detect PD without specialized hardware or controlled tasks
- Temporal patterns (morning vs. evening typing) reveal medication-cycle-driven motor impairments
- Explainable AI (SHAP) provides the clinical transparency needed for medical adoption

---

## 🗂️ Repository Structure

```
parkinsons-detection-keystroke-dynamics/
├── notebooks/
│   ├── data_preparation_eda.ipynb        # Data ingestion, cleaning, feature engineering, EDA
│   └── model_training_xai.ipynb          # Circadian feature extraction, SMOTE, ensemble training, SHAP
├── data/
│   └── processed/
│       └── parkinsons_keystroke_features.csv  # Engineered feature dataset (227 users)
├── assets/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── feature_importance.png
├── reports/
│   └── parkinsons_detection_report.pdf   # Full technical paper with methodology and results
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/parkinsons-detection-keystroke-dynamics.git
cd parkinsons-detection-keystroke-dynamics
pip install -r requirements.txt
```

### Dataset

This project uses the **Tappy Keystroke Dataset** from PhysioNet:
- **227 users** (173 healthy, 54 with Parkinson's Disease)
- **9+ million keystrokes** collected passively over several months
- Available at: [PhysioNet - Tappy Keystroke Data](https://physionet.org/content/tappy/1.0.0/)

To reproduce the analysis:
1. Download the dataset from PhysioNet
2. Place the raw data files in a `Dataset/` directory adjacent to the notebooks
3. Run `data_preparation_eda.ipynb` to generate the processed feature CSV
4. Run `model_training_xai.ipynb` to reproduce model training and SHAP analysis

### Running the Notebooks

```bash
jupyter notebook notebooks/data_preparation_eda.ipynb
jupyter notebook notebooks/model_training_xai.ipynb
```

---

## 🔬 Technical Approach

### 1. **Data Preprocessing**
- Removed keystrokes with HoldTime > 2000ms (non-typing key holds)
- Excluded users with < 50 total keystrokes
- Aggregated time-series data into statistical features per user

### 2. **Feature Engineering**

| Feature Category | Examples | Clinical Relevance |
|---|---|---|
| **Temporal Statistics** | Mean, Std, Skewness, Kurtosis of HoldTime | Captures typing speed degradation |
| **Hand-Specific Metrics** | Left/Right hand HoldTime and Latency | Models unilateral motor impairment |
| **Novel Clinical Features** | **Circadian HoldTime Fluctuation**, HoldTime Asymmetry | Quantifies Levodopa-induced On/Off fluctuations |

**Circadian HoldTime Fluctuation Calculation:**
1. Extract timestamps from raw keystroke logs
2. Bin keystrokes into Morning (6am–12pm) and Evening (6pm–12am)
3. Compute: `|Mean_HoldTime_Morning - Mean_HoldTime_Evening|`
4. Larger differences indicate severe motor fluctuations tied to medication cycles

### 3. **Handling Class Imbalance**
- **Original distribution:** 75% Healthy, 25% PD
- **SMOTE applied to training set** to generate synthetic PD samples
- **Stratified train/test split** to preserve class proportions in evaluation

### 4. **Model Selection: Soft Voting Ensemble**

The ensemble combines three complementary algorithms:
- **Random Forest:** Captures non-linear feature interactions through decision tree ensembles
- **XGBoost:** Gradient boosting optimized for tabular data with regularization
- **SVM (RBF kernel):** Projects data to higher-dimensional space for optimal hyperplane separation

**Soft Voting** averages predicted probabilities rather than class labels, enabling:
- Continuous confidence scores (e.g., "78% probability of PD")
- Reduced overfitting through model diversity
- More nuanced clinical decision support

### 5. **Explainable AI (SHAP)**

SHapley Additive exPlanations decompose each prediction into feature contributions:
- **Global importance:** Which features matter most across all predictions?
- **Instance-level explanations:** Why did the model classify this specific user as PD?
- **Clinical validation:** Confirms predictions are driven by Circadian Fluctuation and Asymmetry, not noise

---

## 📈 Key Findings

1. **Circadian patterns are predictive:** Users with large morning-evening typing speed differences are more likely to have PD, aligning with known Levodopa medication cycles.

2. **Ensemble methods reduce bias:** Combining diverse algorithms improved robustness on unconstrained, noisy data.

3. **Explainability enables trust:** SHAP visualizations confirmed the model learned clinically meaningful patterns, addressing the "black box" problem that hinders medical AI adoption.

4. **Limitations acknowledged:**
   - Small PD sample size (54 users) — results should be validated on larger cohorts
   - SMOTE synthetic data cannot replace real physiological variation
   - Model requires prospective clinical validation before deployment

---

## 🛠️ Tech Stack

- **Machine Learning:** scikit-learn (Random Forest, SVM, GridSearchCV), XGBoost, imbalanced-learn (SMOTE)
- **Explainable AI:** SHAP (TreeExplainer)
- **Data Processing:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Notebook Environment:** Jupyter, IPython

---

## 📄 Research Report

The complete methodology, literature review, experimental setup, results, and limitations are documented in:

**[`reports/parkinsons_detection_report.pdf`](reports/parkinsons_detection_report.pdf)**

This technical paper follows academic conference format and includes:
- Detailed feature engineering rationale
- Hyperparameter tuning grids
- Per-class precision/recall/F1 scores
- SHAP interpretation guidelines
- Comparison to prior work in keystroke-based PD detection

---

## 🔮 Future Work

- **Transformer-based models:** Apply BERT/GPT-style architectures to keystroke sequences
- **Larger datasets:** Validate on multi-site clinical trial data
- **Real-time monitoring:** Build a browser extension for continuous passive monitoring
- **Severity prediction:** Extend from binary classification to UPDRS score regression
- **Longitudinal tracking:** Monitor disease progression over months/years

---

## 📚 References

1. Giancardo, L., et al. (2016). "Computer keyboard interaction as an indicator of early Parkinson's disease." *Scientific Reports*.
2. Lundberg, S. M., & Lee, S.-I. (2017). "A unified approach to interpreting model predictions." *NeurIPS*.
3. Adams, W. R. (2017). "High-accuracy detection of early Parkinson's Disease using multiple characteristics of finger movement while typing." *PLOS ONE*.
4. PhysioNet. (2018). "Tappy Keystroke Data." Available at: https://physionet.org/content/tappy/1.0.0/

---

## 👤 Author

**Fayez Ahmad Protik**

This project demonstrates expertise in:
- Clinical machine learning with domain-specific feature engineering
- Handling imbalanced datasets and evaluating model fairness
- Explainable AI integration for high-stakes applications
- End-to-end ML pipeline from raw data to validated insights

For questions or collaboration opportunities, feel free to connect through my GitHub profile.

---

## 📝 License

This project is provided for educational and research purposes. The Tappy Keystroke Dataset is subject to PhysioNet's usage terms. Please cite the original data source and this repository if you build upon this work.
