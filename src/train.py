import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import find_file
from utils import generate_and_train_intent_model, QA_DATASET_PATH, INTENT_MODEL_PATH

MODEL_OUT = "models/dropout_model.joblib"

def train():
    print("=== 1. HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN BỎ HỌC (RANDOM FOREST) ===")
    x_train_p = find_file("X_train_tree.csv")
    y_train_p = find_file("y_train.csv")
    x_val_p   = find_file("X_validation_tree.csv")
    y_val_p   = find_file("y_validation.csv")
    x_test_p  = find_file("X_test_tree.csv")
    y_test_p  = find_file("y_test.csv")

    if not x_train_p or not y_train_p:
        raise FileNotFoundError("Không tìm thấy X_train_tree.csv hoặc y_train.csv trong thư mục data/!")

    X_train = pd.read_csv(x_train_p)
    y_train = pd.read_csv(y_train_p).iloc[:, -1]
    X_val   = pd.read_csv(x_val_p)
    y_val   = pd.read_csv(y_val_p).iloc[:, -1]
    X_test  = pd.read_csv(x_test_p)
    y_test  = pd.read_csv(y_test_p).iloc[:, -1]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    print(f"Validation ROC-AUC: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.4f}")
    print(f"Test ROC-AUC:       {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": list(X_train.columns)
    }, MODEL_OUT)
    print(f"Đã lưu mô hình dự báo bỏ học tại: {MODEL_OUT}\n")

    print("=== 2. SINH DỮ LIỆU HỎI ĐÁP & HUẤN LUYỆN MÔ HÌNH NLP CHATBOT ===")
    nlp_model = generate_and_train_intent_model()
    df_qa = pd.read_csv(QA_DATASET_PATH)
    print(f"Đã tạo tập dữ liệu Q&A gồm {len(df_qa)} mẫu câu hỏi ({df_qa['intent'].nunique()} chủ đề) tại: {QA_DATASET_PATH}")
    print(f"Đã lưu mô hình phân loại câu hỏi NLP tại: {INTENT_MODEL_PATH}")

if __name__ == "__main__":
    train()
