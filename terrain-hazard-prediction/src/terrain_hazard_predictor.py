import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# ----------------------------------------------------
# 0. LOAD DATA
# ----------------------------------------------------

def load_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "landslide_data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError("ERROR: 'landslide_data.csv' not found in project root.")

    df = pd.read_csv(data_path)
    return df

# ----------------------------------------------------
# 1. TRAIN MODEL
# ----------------------------------------------------

def train_model(df):
    X = df.drop("Landslide", axis=1)
    y = df["Landslide"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))
    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    return clf, X_test, y_test, y_pred

# ----------------------------------------------------
# 2. HEATMAP
# ----------------------------------------------------

def plot_heatmap(df):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap="viridis")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "heatmap.png"), dpi=300)
    plt.close()

# ----------------------------------------------------
# 3. FEATURE IMPORTANCE
# ----------------------------------------------------

def plot_feature_importance(clf, df):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    importance = clf.feature_importances_
    features = df.drop("Landslide", axis=1).columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features)
    plt.title("Feature Importance for Landslide Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=300)
    plt.close()

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()

    print("Training model...")
    clf, X_test, y_test, y_pred = train_model(df)

    print("Generating plots...")
    plot_heatmap(df)
    plot_feature_importance(clf, df)

    print("DONE! Check the 'figures' folder for output images.")
