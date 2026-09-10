def behavioral_scoring(
    debtrules3,
    debtnorms8,
    debtrules16,
    debtpers5,
    debtnorms9,
    debtpers9,
    debtpers11,
    debtrules9
):
    """
    Calculate gamma_short = Debt Aversion, alpha_short = Risk Aversion, lambda_short = Loss Avesion, and delta_short = Time Preference.

    Parameters:
        debtrules3, debtnorms8, debtrules16, debtpers5,
        debtnorms9, debtpers9, debtpers11, debtrules9:
            Numeric input values on scale (1 to 5).

    Returns:
        tuple: (gamma_short, alpha_short, lambda_short, delta_short)
    """

    gamma_short = (
        1.0694
        + 0.0045 * debtrules3
        - 0.0067 * debtnorms8
    )

    alpha_short = (
        0.501796
        + 0.06459 * debtrules16
        - 0.034671 * debtpers5
        - 0.044603 * debtnorms9
    )

    lambda_short = (
        1.177504
        - 0.025919 * debtrules16
        + 0.028538 * debtpers9
    )

    delta_short = (
        0.050555
        - 0.003716 * debtpers9
        + 0.001902 * debtpers11
        - 0.000938 * debtrules9
    )

    return gamma_short, alpha_short, lambda_short, delta_short
