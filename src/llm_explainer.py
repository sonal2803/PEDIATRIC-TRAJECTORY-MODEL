"""
Generative AI interpretation layer (paper Section III.F).

Uses LLaMA-3.3-70B via Groq API to convert probabilistic risk trajectory
outputs into medically grounded narrative explanations without modifying
the underlying numerical predictions.

Prompts are tuned to:
  - Maintain probabilistic (non-deterministic) language
  - Tailor vocabulary to the selected audience (parent vs clinician)
  - Include mechanistic reasoning, uncertainty acknowledgment,
    rare-disease considerations, and an explicit non-diagnostic disclaimer
"""

import os
from groq import Groq


def generate_detailed_explanation(
    disease_input: str,
    domain: str,
    category: str,
    mean_risk: "np.ndarray",
    selected_stage: str,
    explanation_mode: str,
    trend: str,
    variability: float,
) -> str:
    """
    Generate a structured clinical/parent-friendly narrative explanation.

    Args:
        disease_input    : free-text diagnosed condition (may be empty)
        domain           : classified disease domain
        category         : "Low Risk" / "Moderate Risk" / "High Risk"
        mean_risk        : numpy array of mean risk values per stage
        selected_stage   : current developmental stage label
        explanation_mode : "Parent-Friendly" or "Clinical Detail"
        trend            : "rising" / "stable" / "declining"
        variability      : mean standard deviation across simulations

    Returns:
        explanation_text : structured narrative string
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )

    client = Groq(api_key=api_key)

    trajectory_summary = ", ".join([f"{r:.3f}" for r in mean_risk.tolist()])

    if explanation_mode == "Parent-Friendly":
        audience_instruction = (
            "Write in plain, compassionate language that parents or caregivers "
            "can understand without a medical background. Avoid clinical jargon. "
            "Use short sentences and an empathetic tone."
        )
    else:
        audience_instruction = (
            "Write in precise clinical language appropriate for a pediatric "
            "neurologist or neuropediatrician. Use standard medical terminology, "
            "reference the model's probabilistic outputs, and include mechanistic detail."
        )

    condition_text = (
        f"Diagnosed or suspected condition: {disease_input}"
        if disease_input and disease_input.strip()
        else "No specific diagnosis provided; analysis is domain-level only."
    )

    prompt = f"""You are a specialist in pediatric neurodevelopment providing a structured clinical report.

Patient context:
- Current developmental stage: {selected_stage}
- {condition_text}
- Neurological disease domain: {domain or "Unclassified / Structural (default)"}
- Model risk category: {category}
- Risk trajectory values (stage-by-stage): {trajectory_summary}
- Trajectory trend pattern: {trend}
- Average prediction variability (uncertainty): {variability:.3f}

Audience instructions: {audience_instruction}

Please provide a structured response with exactly these five sections:

1. MECHANISTIC EXPLANATION
   Explain the biological or pathophysiological basis of this disease domain and how it
   relates to the observed trajectory pattern.

2. TRAJECTORY INTERPRETATION
   Interpret the {trend} risk pattern across developmental stages. Explain what a
   {trend} pattern typically implies for a child with this domain of disorder.

3. UNCERTAINTY AND VARIABILITY
   Explain what the variability score of {variability:.3f} means in practical terms.
   A score below 0.05 indicates relatively consistent projections; above 0.10 indicates
   high uncertainty requiring close monitoring.

4. RARE DISEASE CONSIDERATIONS
   Highlight any specific challenges, clinical monitoring priorities, or intervention
   windows relevant to rare pediatric neurological conditions in this domain.

5. DISCLAIMER
   Clearly state that this report is based on a probabilistic computational model,
   does not constitute medical diagnosis, and that all clinical decisions must be
   made in consultation with a qualified pediatric neurologist.

IMPORTANT: Do NOT make deterministic prognostic claims. Use language such as
"the model suggests...", "this pattern is consistent with...", "monitoring is advisable
because..." — never "the child will..." or "this confirms a diagnosis of..."."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evidence-based medical reasoning assistant specialising in "
                    "rare pediatric neurological diseases. You communicate probabilistic model "
                    "outputs responsibly and always include non-diagnostic disclaimers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,  # Lower temperature for more consistent clinical output
        max_tokens=1200,
    )

    return response.choices[0].message.content