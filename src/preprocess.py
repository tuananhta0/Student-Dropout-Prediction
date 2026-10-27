import os
import joblib
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/dropout_model.joblib"

def find_file(filename: str, search_dir: str = "data") -> Optional[str]:
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

class StudentDataManager:
    def __init__(self):
        snap_path = find_file("dataset_cleaned.csv") or find_file("hk1_preprocessing_input.csv")
        if not snap_path:
            raise FileNotFoundError("Không tìm thấy dataset_cleaned.csv trong thư mục data/!")
        self.df_raw = pd.read_csv(snap_path)
        self.id_col = self._detect_id_column(self.df_raw)
        self.df_raw[self.id_col] = self.df_raw[self.id_col].astype(str).str.strip()

        self._ensure_model_trained()
        self.model_bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        self.features_lookup = self._build_features_lookup()

    def _ensure_model_trained(self):
        if not os.path.exists(MODEL_PATH):
            x_train_p = find_file("X_train_tree.csv")
            y_train_p = find_file("y_train.csv")
            if x_train_p and y_train_p:
                X_train = pd.read_csv(x_train_p)
                y_train = pd.read_csv(y_train_p).iloc[:, -1]
                model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
                model.fit(X_train, y_train)
                os.makedirs("models", exist_ok=True)
                joblib.dump({"model": model, "feature_columns": list(X_train.columns)}, MODEL_PATH)

    def _detect_id_column(self, df: pd.DataFrame) -> str:
        for candidate in ["student_id", "mssv", "ma_sv", "student_code", "id", "row_key"]:
            for col in df.columns:
                if candidate in col.lower():
                    return col
        return df.columns[0]

    def _build_features_lookup(self) -> Optional[pd.DataFrame]:
        dfs = []
        sample_ids = set(self.df_raw[self.id_col].head(50).values)
        for split in ["train", "validation", "test"]:
            rk_path = find_file(f"row_keys_{split}.csv")
            x_path = find_file(f"X_{split}_tree.csv")
            if rk_path and x_path:
                rk = pd.read_csv(rk_path)
                x = pd.read_csv(x_path)
                # Tìm đúng cột chứa mã sinh viên khớp với dataset_cleaned.csv
                matched_col = None
                for col in rk.columns:
                    vals = set(rk[col].astype(str).str.strip().head(50).values)
                    if vals.intersection(sample_ids):
                        matched_col = col
                        break
                if matched_col is None:
                    matched_col = self._detect_id_column(rk)
                x["_student_key"] = rk[matched_col].astype(str).str.strip().values
                dfs.append(x)
        if dfs:
            full_x = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["_student_key"])
            return full_x.set_index("_student_key")
        return None

    def get_sample_student_ids(self, n: int = 5):
        return self.df_raw[self.id_col].head(n).tolist()

    def authenticate(self, username: str, password: str) -> bool:
        username = username.strip()
        if username not in self.df_raw[self.id_col].values:
            return False
        return password == username or password == "cmc@123"

    def get_personal_profile_and_prediction(self, student_id: str) -> Dict[str, Any]:
        student_id = student_id.strip()
        row = self.df_raw[self.df_raw[self.id_col] == student_id]
        if row.empty:
            return {}
        profile = row.iloc[0].to_dict()
        dropout_prob = None

        if self.model_bundle and self.features_lookup is not None:
            model = self.model_bundle["model"]
            feat_cols = self.model_bundle["feature_columns"]
            if student_id in self.features_lookup.index:
                student_X = self.features_lookup.loc[[student_id], feat_cols]
                dropout_prob = float(model.predict_proba(student_X)[0, 1])
            else:
                # Khớp theo vị trí dòng nếu row_keys lưu theo index số
                idx = int(row.index[0])
                if str(idx) in self.features_lookup.index:
                    student_X = self.features_lookup.loc[[str(idx)], feat_cols]
                    dropout_prob = float(model.predict_proba(student_X)[0, 1])
                elif idx < len(self.features_lookup):
                    student_X = self.features_lookup.iloc[[idx]][feat_cols]
                    dropout_prob = float(model.predict_proba(student_X)[0, 1])

        return {
            "student_id": student_id,
            "profile": profile,
            "dropout_probability": dropout_prob
        }
