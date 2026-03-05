from groq import Groq
import os


def generate_detailed_explanation(
    disease_input,
    domain,
    category,
    mean_risk,
    selected_stage,
    explanation_mode,
    trend,
    variability
):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    client = Groq(api_key=api_key)

    trajectory_summary = ", ".join([f"{r:.3f}" for r in mean_risk.tolist()])
    audience = "parents" if explanation_mode == "Parent-Friendly" else "clinicians"

    prompt = f"""
You are a pediatric neurodevelopment specialist.

Stage: {selected_stage}
Condition: {disease_input or "None provided"}
Domain: {domain or "Unclassified"}
Risk Category: {category}

Trajectory values:
{trajectory_summary}

Trend pattern: {trend}
Average variability: {variability:.3f}

Provide:
1. Mechanistic explanation
2. Clinical interpretation of a {trend} pattern
3. Interpretation of uncertainty (variability)
4. Rare disease considerations if applicable
5. Clear disclaimer of probabilistic modeling

Audience: {audience}
Avoid deterministic claims.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an evidence-based medical reasoning assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content