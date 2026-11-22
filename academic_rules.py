"""Academic rules for QĐ 22/2023 (simulation, not official decisions)."""
import math
from decimal import Decimal, ROUND_HALF_UP

GRADE_POINTS = {'A+': 4, 'A': 4, 'B+': 3.5, 'B': 3, 'C+': 2.5,
                'C': 2, 'D+': 1.5, 'D': 1, 'F': 0}


def number(value, low=0, high=math.inf):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) and low <= n <= high else None


def rounded(value, digits=2):
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def registration_bounds(standard):
    n = number(standard)
    if n is None or n == 0:
        raise ValueError('Positive standard load required')
    return math.ceil(2*n/3), math.floor(3*n/2)


def warning_reasons(term, gpa, registered, failed, owed):
    """Article 18.1: owed excludes certificate courses. None means unknown."""
    term, gpa = number(term, 1), number(gpa, 0, 4)
    registered, failed, owed = number(registered), number(failed), number(owed)
    if registered is not None and failed is not None and failed > registered:
        raise ValueError('Failed credits exceed registered credits')
    return {
        'gpa': None if term is None or gpa is None else rounded(gpa) < (.8 if term == 1 else 1),
        'failed': None if registered is None or failed is None else 2*failed > registered,
        'owed': None if owed is None else owed > 24,
    }


def withdrawal_reasons(streak, total, duration, degree):
    streak, total, duration = number(streak), number(total), number(duration)
    limit = {'Bachelor': 18, 'Engineer': 22}.get(degree)
    return {'consecutive': None if streak is None else streak > 2,
            'total': None if total is None else total > 3,
            'duration': None if duration is None or limit is None else duration > limit}


def exam_barred(scheduled, unexcused, excused):
    """Article 25.1a: per course, not average attendance."""
    scheduled, unexcused, excused = map(number, (scheduled, unexcused, excused))
    if None in (scheduled, unexcused, excused):
        return None
    if scheduled == 0 or unexcused + excused > scheduled:
        raise ValueError('Invalid attendance hours')
    return unexcused * 5 > scheduled or (unexcused + excused) * 10 > 3 * scheduled


def course_grade(final_score, exam_score, *, barred=False, retest=False, gpa_eligible=True):
    """Art. 13–15. Retest (thi lại) differs from a new course attempt."""
    if barred is True:
        return 'F'
    score, exam = number(final_score, 0, 10), number(exam_score, 0, 10)
    if None in (score, exam) or barred is None:
        return None
    if exam <= 2:
        return 'F'
    score = rounded(score, 1)
    if not gpa_eligible:
        # Scores 4.0–4.9 do not satisfy the P threshold; ask for official result.
        return 'P' if score >= 5 else 'F' if score < 4 else None
    if retest:
        score = min(score, 6.4)
    return next(grade for cutoff, grade in [(9,'A+'),(8.5,'A'),(8,'B+'),(7,'B'),
        (6.5,'C+'),(5.5,'C'),(5,'D+'),(4,'D'),(0,'F')] if score >= cutoff)


def official_gpa(attempts):
    """Best completed GPA grade per course; F contributes zero points and credits.
    Each input row: course_id, registered_credits, grade_letter, is_gpa_eligible.
    Non-GPA and I/X/R/P excluded. No silent assumption for missing eligibility.
    """
    best = {}
    for a in attempts:
        if a.get('is_gpa_eligible') is None:
            raise ValueError('Missing GPA eligibility')
        if not a['is_gpa_eligible'] or a.get('is_certificate_course', False):
            continue
        grade = a.get('grade_letter')
        if grade not in GRADE_POINTS:
            continue
        credits = number(a.get('registered_credits'), 1)
        if credits is None:
            raise ValueError('Missing course credits')
        cid, points = a['course_id'], GRADE_POINTS[grade]
        if cid in best and best[cid][0] != credits:
            raise ValueError('Credits changed; official equivalency required')
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
    """Illustrative new-course-only GPA scenarios (not a trained forecast)."""
    pts = number(row.get('gpa_quality_points'))
    credits = number(row.get('gpa_credit_count'))
    gpa = number(row.get('gpa4_cumulative'), 0, 4)
    term = number(row.get('snapshot_term_index'), 1)
    remaining = number(row.get('credits_required'))
    earned = number(row.get('credits_earned_cumulative'))
    if None in (pts, credits, gpa, term, remaining, earned) or pts > 4*credits:
        return None
    n = min(15, max(0, remaining-earned))
    if n == 0:
        return None
    # Explicit scenarios, no fabricated behavioral forecast.
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
    """Art. 19–20. Unknown conditions cannot be marked passed."""
    earned, required = number(row.get('credits_earned_cumulative')), number(row.get('credits_required'))
    gpa, debt = number(row.get('gpa4_cumulative'),0,4), number(row.get('tuition_debt_end_vnd'))
    certs = set(str(row.get('verified_certificates') or '').replace(' ','').split(','))
    remaining = number(row.get('mandatory_courses_remaining'))
    return {'credits': None if None in (earned,required) else earned >= required,
            'mandatory_courses': None if remaining is None else remaining == 0,
            'gpa': None if gpa is None else rounded(gpa) >= 2,
            'english': True if 'ENGLISH' in certs else None,
            'physical_education': True if 'PE' in certs else None,
            'national_defense': True if 'DEFENSE' in certs else None,
            'financial': False if debt is not None and debt > 0 else None,
            'program_outcomes': row.get('program_outcomes_verified'),
            'discipline_and_criminal_clearance': row.get('discipline_and_criminal_clearance')}
