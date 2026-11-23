"""Chuẩn hóa đầu vào, phân hạng và kịch bản ML không suy diễn nhân quả."""
import numpy as np
import pandas as pd


def model_inputs(frame, meta):
    cols = meta['feature_cols']
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f'Thiếu đặc trưng mô hình: {sorted(missing)}')
    out = frame.loc[:, cols].copy()
    for c in meta['cat_cols']:
        if meta.get('categorical_missing_token'):
            out[c] = out[c].astype(object).map(lambda v: meta['categorical_missing_token'] if pd.isna(v) else str(v))
        else:
            # Tương thích artifact cũ; không dùng quy tắc này cho mô hình mới.
            out[c] = out[c].astype(str)
    for c in meta.get('num_cols', []):
        out[c] = pd.to_numeric(out[c], errors='raise').astype(float)
        if np.isinf(out[c]).any():
            raise ValueError(f'Đặc trưng vô hạn: {c}')
    return out


def risk_level(probability, threshold):
    if probability >= max(.45, threshold):
        return 'Rất cao'
    if probability >= threshold:
        return 'Cao'
    return 'Trung bình' if probability >= min(.12, threshold*.8) else 'Thấp'


def apply_prediction_scope(frame, meta):
    """Không coi xác suất trước thời điểm huấn luyện hoặc ngoài miền HK là can thiệp hợp lệ."""
    if 'model_available_as_of' not in meta:
        return frame
    out = frame.copy()
    valid = out.prediction_status.eq('VALID')
    supported = out.prediction_term.isin(meta['supported_terms'])
    before = out.prediction_cutoff.fillna('').lt(meta['model_available_as_of'])
    out.loc[valid & ~supported, 'prediction_status'] = 'OUT_OF_SCOPE'
    out.loc[valid & supported & before, 'prediction_status'] = 'RETROSPECTIVE'
    return out


def payment_scenario(snapshot, payment):
    """Giả định thanh toán ở cuối kỳ, không thay GPA/chuyên cần hay lịch sử bất liên quan."""
    if isinstance(payment, (bool, np.bool_)):
        raise ValueError('Số tiền không hợp lệ')
    debt = float(snapshot['tuition_debt_end_vnd'])
    ratio = float(snapshot['tuition_debt_to_due_ratio'])
    days = float(snapshot['tuition_overdue_days_end'])
    count = float(snapshot['tuition_overdue_terms_to_date'])
    payment = float(payment)
    if not all(np.isfinite(v) for v in (debt, ratio, days, count, payment)) or not 0 < payment <= debt or ratio <= 0 or days < 0 or count < 0:
        raise ValueError('Cần số tiền dương không vượt công nợ và đủ dữ liệu tài chính')
    if days > 0 and count < 1:
        raise ValueError('Lịch sử quá hạn không nhất quán')
    out = dict(snapshot)
    new_debt = max(0., debt-payment)
    out['tuition_debt_end_vnd'] = new_debt
    out['tuition_debt_to_due_ratio'] = ratio*new_debt/debt
    # Giả định thanh toán trước khi chốt cùng kỳ; quy tắc bộ sinh ghi quá hạn khi nợ > 500.000đ.
    if new_debt <= 500000:
        out['tuition_overdue_days_end'] = 0
        out['tuition_overdue_terms_to_date'] = count-int(days>0)
    return out
