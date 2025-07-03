# Project-AI-and-Machine-Learning-uom-2025

# 📊 Bankruptcy Prediction using Machine Learning 

Addresses the problem of predicting whether a company will go bankrupt based on financial ratios using several classification techniques.

---

## 📁 Structure

The repository includes:

- 📂 `project2company/`: All core Python scripts, data preprocessing, model training and evaluation.
- 📄 ΕφΠλη_Εργασία 2.pdf: Final report (Assignment 2).
- 📄 ΕφΠλη_Εργασία 3.pdf: Follow-up report.
- 📄 Καλές πρακτικές checklist.pdf: Report writing guidelines.

---

## ⚙️ Tools & Technologies

- **Python 3.x**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn** (MinMaxScaler, StratifiedKFold, classifiers)
- **Google Colab** (execution environment)
- **Excel** (for pivot table visualization)

---

## 🔍 Main Steps of the Pipeline

1. **Data Validation** – Check for missing values (NaN)
2. **Normalization** – Apply MinMax scaling to numeric features
3. **Stratified K-Fold Split** (k=4)
4. **Downsampling** – Balance classes (3:1 ratio of healthy to bankrupt)
5. **Model Training & Evaluation** on 8 classifiers:
   - Logistic Regression, LDA, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting
6. **Performance Metrics** – Accuracy, Precision, Recall, F1, AUC, Recall_Healthy
7. **Confusion Matrix** – Train/Test visualized
8. **Export** to `.csv` for further Excel-based analysis

---

## 📊 Results & Visualization

- Output metrics are saved to: `balancedDataOutcomes.csv`
- Excel pivot tables were used to compare average performance across classifiers.
- Visual comparisons: stacked bar charts, F1 vs Recall, grouped charts.

---

## ▶️ How to Run

### Google Colab (recommended)

1. Open any `.py` script from this repository in [Google Colab](https://colab.research.google.com/)
2. Run the notebook cells sequentially.
3. Outputs will appear inline (metrics, confusion matrices, graphs).

Alternatively, download the repo and run locally with:

```bash
pip install -r requirements.txt
python check_and_normalize.py
python model_loop_all.py
