from typing import Dict

def calculate_financial_wellbeing(
    dti: float,
    cash_buffer_months: float,
    savings_rate: float,
    late_frequency: float,
    high_interest_debt_share: float,
    source: str = "mixed"               # optional: "statement", "conversation", or "mixed"
) -> Dict:
    """
    Simple rule-based Financial Well-Being Score (0-100)
    Features are designed to be collectible from both bank statements and conversation.
    """

    # ---------- Individual scores (0-20 each) ----------

    # 1. DTI
    if dti <= 0.20:
        dti_score = 20
    elif dti <= 0.30:
        dti_score = 16
    elif dti <= 0.40:
        dti_score = 12
    elif dti <= 0.50:
        dti_score = 6
    else:
        dti_score = 0

    # 2. Cash Buffer (months)
    if cash_buffer_months >= 3.0:
        buffer_score = 20
    elif cash_buffer_months >= 2.0:
        buffer_score = 15
    elif cash_buffer_months >= 1.0:
        buffer_score = 10
    elif cash_buffer_months >= 0.5:
        buffer_score = 5
    else:
        buffer_score = 0

    # 3. Savings Rate
    if savings_rate >= 0.20:
        savings_score = 20
    elif savings_rate >= 0.10:
        savings_score = 15
    elif savings_rate >= 0.05:
        savings_score = 10
    elif savings_rate >= 0.0:
        savings_score = 5
    else:
        savings_score = 0

    # 4. Late / Overdraft Frequency (last 6 months normalized)
    if late_frequency <= 0.0:
        late_score = 20
    elif late_frequency <= 0.15:          # ≤ 1 time in 6 months
        late_score = 12
    elif late_frequency <= 0.35:          # ≤ 2 times
        late_score = 6
    else:
        late_score = 0

    # 5. High-Interest Debt Share
    if high_interest_debt_share <= 0.20:
        high_int_score = 20
    elif high_interest_debt_share <= 0.40:
        high_int_score = 14
    elif high_interest_debt_share <= 0.60:
        high_int_score = 8
    else:
        high_int_score = 0

    # ---------- Total & Category ----------
    total_score = dti_score + buffer_score + savings_score + late_score + high_int_score

    if total_score >= 80:
        category = "Healthy"
    elif total_score >= 60:
        category = "Moderate"
    elif total_score >= 40:
        category = "Stressed"
    else:
        category = "High Risk"

    return {
        "financial_wellbeing_score": total_score,
        "category": category,
        "breakdown": {
            "dti_score": dti_score,
            "cash_buffer_score": buffer_score,
            "savings_rate_score": savings_score,
            "late_frequency_score": late_score,
            "high_interest_debt_score": high_int_score
        },
        "inputs": {
            "dti": dti,
            "cash_buffer_months": cash_buffer_months,
            "savings_rate": savings_rate,
            "late_frequency": late_frequency,
            "high_interest_debt_share": high_interest_debt_share,
            "source": source
        }
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example: values coming from mixed sources
    result = calculate_financial_wellbeing(
        dti=0.34,                     # from statement or conversation
        cash_buffer_months=1.6,       # from statement or user estimate
        savings_rate=0.07,            # from statement or user estimate
        late_frequency=0.10,          # from statement or user estimate
        high_interest_debt_share=0.45,# from statement or user estimate
        source="mixed"
    )

    print("Score:", result["financial_wellbeing_score"])
    print("Category:", result["category"])
    print("\nBreakdown:")
    for k, v in result["breakdown"].items():
        print(f"  {k}: {v}")