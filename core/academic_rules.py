"""Quy tắc học vụ theo QĐ 22/2023 (mô phỏng, không phải quyết định chính thức)."""
import math
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

GRADE_POINTS = {'A+': 4, 'A': 4, 'B+': 3.5, 'B': 3, 'C+': 2.5,
                'C': 2, 'D+': 1.5, 'D': 1, 'F': 0}


def number(value, low=0, high=math.inf):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) and low <= n <= high else None


def strict_bool(value):
    """Chỉ chấp nhận Boolean thật; ô thiếu là chưa biết."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return None
    raise ValueError('Giá trị luận lý không hợp lệ')


def whole(value, low=0):
    n = number(value, low)
    return n if n is not None and n.is_integer() else None


def rounded(value, digits=2):
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def registration_bounds(standard):
    n = number(standard)
    if n is None or n == 0:
        raise ValueError('Khối lượng chuẩn phải dương')
    return math.ceil(2*n/3), math.floor(3*n/2)


def warning_reasons(term, gpa, registered, failed, owed):
    """Điều 18.1: tín chỉ nợ không gồm môn chứng chỉ. None là chưa xác định."""
    term, gpa = whole(term, 1), number(gpa, 0, 4)
    registered, failed, owed = whole(registered), whole(failed), whole(owed)
    if registered is not None and failed is not None and failed > registered:
        raise ValueError('Tín chỉ trượt vượt số tín chỉ đăng ký')
    return {
        'gpa': None if term is None or gpa is None else rounded(gpa) < (.8 if term == 1 else 1),
        'failed': None if registered is None or failed is None else 2*failed > registered,
        'owed': None if owed is None else owed > 24,
    }


def withdrawal_reasons(streak, total, duration, degree):
    streak, total, duration = whole(streak), whole(total), whole(duration)
    if streak is not None and total is not None and streak > total:
        raise ValueError("Số lần cảnh báo liên tiếp vượt tổng số lần")
    limit = {'Bachelor': 18, 'Engineer': 22}.get(degree)
    return {'consecutive': None if streak is None else streak > 2,
            'total': None if total is None else total > 3,
            'duration': None if duration is None or limit is None else duration > limit}


def exam_barred(scheduled, unexcused, excused):
    """Điều 25.1a: tính theo từng học phần."""
    scheduled, unexcused, excused = map(number, (scheduled, unexcused, excused))
    if None in (scheduled, unexcused, excused):
        return None
    if scheduled == 0 or unexcused + excused > scheduled:
        raise ValueError('Số giờ vắng không hợp lệ')
    return unexcused * 5 > scheduled or (unexcused + excused) * 10 > 3 * scheduled


def course_grade(final_score, exam_score, *, barred=False, retest=False, gpa_eligible=True):
    """Điều 13–15. Phân biệt thi lại và học lại."""
    barred, retest, gpa_eligible = map(strict_bool, (barred,retest,gpa_eligible))
    if None in (barred,retest,gpa_eligible):
        return None
    if barred is True:
        return 'F'
    score, exam = number(final_score, 0, 10), number(exam_score, 0, 10)
    if None in (score, exam) or barred is None:
        return None
    if exam <= 2:
        return 'F'
    score = rounded(score, 1)
    if not gpa_eligible:
        # Điểm 4,0–4,9 chưa đạt ngưỡng P; cần kết quả chính thức.
        return 'P' if score >= 5 else 'F' if score < 4 else None
    if retest:
        score = min(score, 6.4)
    return next(grade for cutoff, grade in [(9,'A+'),(8.5,'A'),(8,'B+'),(7,'B'),
        (6.5,'C+'),(5.5,'C'),(5,'D+'),(4,'D'),(0,'F')] if score >= cutoff)


def official_gpa(attempts):
    """Lấy điểm chính thức cao nhất theo học phần. Điểm F bằng 0 nhưng vẫn tính tín chỉ.
    Loại I/X/R/P và môn không tính GPA; không tự giả định khi thiếu dữ liệu.
    """
    best = {}
    for a in attempts:
        eligible = strict_bool(a.get('is_gpa_eligible'))
        certificate = strict_bool(a.get('is_certificate_course', False))
        if None in (eligible,certificate):
            raise ValueError('Thiếu điều kiện tính GPA')
        if not eligible or certificate:
            continue
        grade = a.get('grade_letter')
        if not isinstance(grade,str) or grade not in GRADE_POINTS:
            if isinstance(grade,str) and grade in ('I','X','R','P'):
                continue
            raise ValueError('Điểm chữ chưa xác định hoặc không hợp lệ')
        credits = number(a.get('registered_credits'), 1)
        if credits is None:
            raise ValueError('Thiếu tín chỉ học phần')
        cid, points = a['course_id'], GRADE_POINTS[grade]
        if cid in best and best[cid][0] != credits:
            raise ValueError('Tín chỉ thay đổi; cần xác nhận tương đương học phần')
        best[cid] = (credits, max(points, best.get(cid, (credits, 0))[1]))
    credits = sum(c for c, _ in best.values())
    points = sum(c*p for c, p in best.values())
    return {'credits': credits, 'quality_points': points,
            'gpa': rounded(points/credits) if credits else None}


def projected_gpa(points, credits, new_gpa, new_credits):
    points, credits, new_gpa, new_credits = number(points), number(credits), number(new_gpa,0,4), number(new_credits)
    if None in (points, credits, new_gpa, new_credits) or points > 4*credits:
        return None
    return rounded((points+new_gpa*new_credits)/(credits+new_credits)) if credits+new_credits else None


def projection(row):
    """Kịch bản GPA chỉ học môn mới, không phải mô hình dự báo."""
    pts = number(row.get('gpa_quality_points'))
    credits = number(row.get('gpa_credit_count'))
    gpa = number(row.get('gpa4_cumulative'), 0, 4)
    term = whole(row.get('snapshot_term_index'), 1)
    remaining = number(row.get('credits_required'))
    earned = number(row.get('credits_earned_cumulative'))
    if None in (pts, credits, gpa, term, remaining, earned) or pts > 4*credits:
        return None
    available = number(row.get('new_course_credits_available'))
    cap = number(row.get('registration_max_credits'))
    owed=number(row.get("credits_owed_cumulative"))
    budget=max(0,remaining-earned-(owed or 0))
    n = min(15, budget, budget if available is None else available, 15 if cap is None else cap)
    if n == 0:
        return None
    # Giả định rõ ràng, không tự suy đoán hành vi.
    gpa = rounded(gpa)
    cur = gpa
    improved = min(4, rounded(cur+.5))
    target = 2 if gpa < 2 else 2.5 if gpa < 2.5 else 3.2 if gpa < 3.2 else gpa
    return {'next_term': int(term)+1, 'next_tc': n, 'gpa_credit_count': credits,
            'gpa_quality_points': pts, 'pred_gpa_cur': rounded(cur),
            'pred_gpa_imp': rounded(improved), 'pred_cum_cur': projected_gpa(pts,credits,cur,n),
            'pred_cum_imp': projected_gpa(pts,credits,improved,n), 'target_cum': target,
            'req_gpa_next': max(0, math.ceil((target*(credits+n)-pts)/n*100-1e-10)/100),
            'assumption': 'NEW_CREDITS_ONLY'}


def graduation_conditions(row):
    """Điều 19–20. Điều kiện chưa rõ không được coi là đạt."""
    earned, required = number(row.get('credits_earned_cumulative')), number(row.get('credits_required'))
    gpa, debt = number(row.get('gpa4_cumulative'),0,4), number(row.get('tuition_debt_end_vnd'))
    certs = set(str(row.get('verified_certificates') or '').replace(' ','').split(','))
    remaining = number(row.get('mandatory_courses_remaining'))
    result = {'credits': None if None in (earned,required) else earned >= required,
            'mandatory_courses': None if remaining is None else remaining == 0,
            'gpa': None if gpa is None else rounded(gpa) >= 2,
            'english': True if 'ENGLISH' in certs else None,
            'physical_education': True if 'PE' in certs else None,
            'national_defense': True if 'DEFENSE' in certs else None,
            'financial': False if debt is not None and debt > 0 else None,
            'program_outcomes': row.get('program_outcomes_verified'),
            'discipline_and_criminal_clearance': row.get('discipline_and_criminal_clearance')}

    # Kết quả xét tốt nghiệp theo hồ sơ nguồn, không thay thế quyết định chính thức.
    mapping = {'credits':'clear_credits','gpa':'clear_gpa4','english':'clear_english',
               'physical_education':'clear_pe','national_defense':'clear_defense',
               'financial':'clear_finance','discipline_and_criminal_clearance':'clear_discipline',
               'academic_debt':'clear_no_outstanding_academic_debt'}
    for key, field in mapping.items():
        recorded = strict_bool(row.get(field))
        if recorded is not None:
            # Bằng chứng không đạt không bị giá trị đạt cũ ghi đè.
            result[key] = False if result.get(key) is False else recorded
    owed = number(row.get('credits_owed_cumulative'))
    if owed is not None and owed > 0:
        result['academic_debt'] = False
    result.setdefault('academic_debt',None)
    result['program_outcomes'] = strict_bool(result['program_outcomes'])
    result['discipline_and_criminal_clearance'] = strict_bool(result['discipline_and_criminal_clearance'])
    return result


def retake_projection(attempts, changes, new_gpa=0, new_credits=0):
    """Tính lại GPA khi học lại/cải thiện các môn đã có điểm chính thức.
    changes là ánh xạ mã học phần -> điểm chữ giả định. Không cộng trùng tín chỉ.
    """
    base = official_gpa(attempts)
    if whole(new_credits) is None or number(new_gpa,0,4) is None:
        raise ValueError("Tín chỉ hoặc GPA giả định không hợp lệ")
    if not isinstance(changes,dict):
        raise ValueError('Danh sách học lại không hợp lệ')
    updated = list(attempts)
    for cid, grade in changes.items():
        matches = [a for a in attempts if a['course_id']==cid and strict_bool(a.get('is_gpa_eligible')) is True and strict_bool(a.get('is_certificate_course',False)) is False and isinstance(a.get('grade_letter'),str) and a['grade_letter'] in GRADE_POINTS]
        if not matches or grade not in GRADE_POINTS:
            raise ValueError('Môn học hoặc điểm chữ học lại không hợp lệ')
        updated.append(dict(matches[0],grade_letter=grade))
    improved = official_gpa(updated)
    final = projected_gpa(improved['quality_points'],improved['credits'],new_gpa,new_credits)
    if final is None:
        raise ValueError('Kịch bản GPA không hợp lệ')
    return {'gpa_before':base['gpa'],'gpa_after':final,'gpa_credits':improved['credits']+float(new_credits),'quality_points':improved['quality_points']+float(new_gpa)*float(new_credits)}
