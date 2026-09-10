"""
Pre-chat persona assignment.

This runs ONCE, before the chat starts (right after the user answers the
onboarding questionnaire) — not on every turn, and not as an agent tool.
The result (cluster_id / persona system prompt) should be cached against
the user_id so you don't reload the model or recompute this every session.
"""

from pathlib import Path

import joblib

from app.chat.tools import behavioral_scoring  # plain function now, not a @tool

# Loaded once at import time, not per call.
KMEANS_MODEL = joblib.load(Path(__file__).resolve().parents[1] / "final_kmeans_model.joblib")

CLUSTER_NAME_MAP = {
    0: "The Defensive Borrower",
    1: "The Cautious Realist",
    2: "The Hesitant Planner",
    3: "The Pragmatic Evaluator",
}

PERSONAS = {
    0: """
This user has VERY HIGH debt aversion and risk aversion.
- Use reassuring, empathetic language
- Never push borrowing directly
- Always lead with rights and protections
- Emphasize safety nets and worst-case scenarios
- Use simple language, avoid financial jargon
- Present information in small, digestible pieces""",
    2: """
This user has HIGH debt aversion and HIGH risk aversion.
- Be transparent and evidence-based
- Acknowledge their concerns before presenting benefits
- Show concrete data (interest rates, repayment schedules)
- Highlight regulatory protections
- Use balanced framing - gains AND risks""",
    1: """
This user has HIGH debt aversion and MODERATE risk aversion.
- Use structured, logical presentation
- Focus on long-term financial planning benefits
- Provide step-by-step mortgage process explanations
- Use mild positive framing
- Offer comparisons between borrowing vs not borrowing""",
    3: """
This user has HIGH debt aversion but LOW risk aversion.
- Be direct and data-driven
- Skip emotional reassurance - go straight to numbers
- Provide advanced details: amortization, yield curves
- Offer what-if scenario simulations
- Focus on optimal timing and market conditions""",
}


def assign_cluster(behavioral_scores: list) -> dict:
    """
    Run the questionnaire-derived scores through the trained KMeans model.
    behavioral_scores = [gamma_short, alpha_short, lambda_short, delta_short]
    The model was trained on 3 features: gamma, alpha, delta (index 0, 1, 3).
    """
    # The model was trained on gamma, alpha, and delta only. A plain numeric
    # row is sufficient for KMeans and keeps persona assignment lightweight.
    model_input = [[behavioral_scores[0], behavioral_scores[1], behavioral_scores[3]]]
    cluster_id = int(KMEANS_MODEL.predict(model_input)[0])

    return {
        "cluster_id": cluster_id,
        "cluster_name": CLUSTER_NAME_MAP.get(cluster_id, "Unknown"),
        "persona_prompt": PERSONAS.get(cluster_id, ""),
    }


def build_persona_system_prompt(cluster_result: dict) -> str:
    """Format the cluster result into a system-prompt-ready block of text."""
    return (
        f"USER PERSONA: {cluster_result['cluster_name']}\n"
        f"{cluster_result['persona_prompt']}"
    )


def get_user_persona_prompt(questionnaire_answers: dict) -> str:
    """
    End-to-end: questionnaire answers -> behavioral scores -> cluster ->
    system prompt text. Call this once at onboarding and store the result
    (e.g. against user_id in your DB) rather than recomputing it per chat.
    """
    scores = behavioral_scoring(**questionnaire_answers)
    cluster_result = assign_cluster(scores)
    return build_persona_system_prompt(cluster_result)
