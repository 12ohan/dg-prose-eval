"""Shared prompt set for the DiffusionGemma vs Gemma-4-AR prose eval.

Design notes
------------
Every prompt is OPEN-ENDED prose generation. No verifiable answers, no math,
no code. Rationale: DiffusionGemma's published benchmark suite is entirely
verifiable tasks (GPQA, AIME, LiveCodeBench, GSM8K) where it loses ground to
the AR baseline. If bidirectional block attention helps anywhere, it should
help where the output has no single correct answer and the constraint is
global consistency rather than left-to-right derivation.

Prompt categories are tagged so we can report per-category deltas rather than
one muddy aggregate. `repetition` prompts are the direct test of the core
hypothesis: AR decoding is structurally prone to elevated n-gram repetition
because every token is conditioned on all prior tokens including previously
repeated ones. A bidirectional canvas can see the repetition within its
256-token window and route around it.
"""

CATEGORIES = {
    "fiction": "Narrative / descriptive prose, tone and voice",
    "repetition": "Long-form where lexical repetition is the failure mode",
    "exposition": "Explaining something precisely without sounding canned",
    "descriptive": "Sensory / observational, diction-heavy",
    "dialogue": "Natural-sounding speech",
}

PROMPTS = [
    # --- fiction: voice, tone, pacing ---
    ("fiction", "Write about 200 words on a train that never arrives at its destination. Hold the tone steady throughout."),
    ("fiction", "Write a short scene, roughly 200 words, of someone packing a suitcase the morning after they decided to leave. No explanation of why."),
    ("fiction", "Write 200 words of literary nonfiction about a swimming pool in winter. Avoid the words 'nostalgic' and 'serene'."),
    ("fiction", "Write about 200 words describing a city street at 4am from the point of view of someone who has just given up on something."),
    ("fiction", "Write 200 words of fiction about a room that has been carefully arranged for a guest who is not coming."),
    ("fiction", "Write 200 words. A character is lying to someone who already knows the truth, and the reader can tell. Do not explain the lie."),

    # --- repetition: the core hypothesis ---
    ("repetition", "Write about 300 words on the difficulty of learning an instrument as an adult."),
    ("repetition", "Write 300 words about a specific neighborhood you have walked through many times."),
    ("repetition", "Write 300 words explaining how a lock works, aimed at a curious adult."),
    ("repetition", "Write 300 words about the appeal of very old books."),
    ("repetition", "Write 300 words on a skill you are trying to build after failing at it repeatedly."),
    ("repetition", "Write 300 words describing the experience of waiting for test results."),
    ("repetition", "Write 300 words about learning to cook a dish from someone else's culture."),
    ("repetition", "Write 300 words on the physical sensation of being tired."),

    # --- exposition: precision without canned phrasing ---
    ("exposition", "Explain in about 250 words how a bill becomes law, for a reader who follows politics but has no legal background."),
    ("exposition", "Explain in about 250 words what makes bread rise, for someone who has never baked."),
    ("exposition", "Explain in about 250 words why the sky appears red at sunset, without using the words 'Rayleigh scattering' or 'wavelength'."),
    ("exposition", "Explain in about 250 words what a compiler does, for a curious non-programmer. Avoid jargon."),
    ("exposition", "Explain in about 250 words how noise-cancelling headphones work, avoiding the phrase 'constructive interference'."),
    ("exposition", "Describe in about 250 words how city water gets treated before reaching your tap."),

    # --- descriptive: diction, sensory specificity ---
    ("descriptive", "Describe the smell of a print shop in about 200 words. Be specific. Do not open with the word 'nostalgic'."),
    ("descriptive", "Write 200 words about the sound of a room after everyone has left."),
    ("descriptive", "Describe in 200 words the quality of light in an early-morning kitchen."),
    ("descriptive", "Write 200 words about the texture of different kinds of paper."),
    ("descriptive", "Describe in 200 words what a city sounds like from inside a moving car."),

    # --- dialogue: natural speech, distinct voices ---
    ("dialogue", "Write 200 words of dialogue between two people who are arguing about something trivial and both know the other is right."),
    ("dialogue", "Write 200 words of dialogue in which one person is lying politely and the other is too polite to press."),
    ("dialogue", "Write 200 words of dialogue between a person and a barista who have known each other for years."),
    ("dialogue", "Write 200 words of dialogue in which someone tries to end a conversation they clearly want to continue."),
]


def get_prompts(category=None):
    if category is None:
        return [(i, c, p) for i, (c, p) in enumerate(PROMPTS)]
    return [(i, c, p) for i, (c, p) in enumerate(PROMPTS) if c == category]


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(PROMPTS)} prompts")
    for cat, n in Counter(c for c, _ in PROMPTS).items():
        print(f"  {cat:14s} {n:3d}  - {CATEGORIES[cat]}")
