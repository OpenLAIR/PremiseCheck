"""Prompts and decision-model questions of PremiseCheck, verbatim from the experiments of the paper
(Appendix B: Tables "Questions given to the decision model" and "The 16 phrasings of the verification question")."""

# ----------------------------------------------------------------------------- decomposer (LLM) -----
# System prompts. The user message is DECOMP_USER with the review comment.
DECOMP_NLP = """You reconstruct ONE peer-review item as an argument, so that each step can be fact-checked separately.

List the premises the item relies on, explicit and implicit, and its conclusion.
- Each premise is ONE self-contained declarative sentence that is either true or false.
- Write it so it can be understood without the review item: say "the paper", and name the method, dataset, table, figure or equation it is about.
- Include premises about what the paper states, reports, shows or omits, and the general-knowledge premises that the reasoning needs.
- Keep each premise as close to the reviewer's own claim as possible. Do not add facts the reviewer did not assert or imply. Do not judge whether a premise is true.
- If the item is a question, write the presuppositions of the question as premises (for example what the reviewer assumes the paper does or does not state).
- Use 1 to 6 premises.

Return ONLY JSON: {"premises": ["...", "..."], "conclusion": "..."}"""

# Machine learning reviews: fewer premises.
DECOMP_ML = """You reconstruct ONE peer-review item as a short argument, so that each step can be fact-checked separately.

List ONLY the premises the reviewer actually asserts or directly presupposes, in the reviewer's own terms:
- the reviewer's claims about what the paper states, reports, shows or omits;
- at most ONE general-knowledge premise, and only if the reviewer's conclusion depends on it.
Each premise is ONE self-contained declarative sentence: say "the paper" and name the method, dataset, table, figure or equation it is about.
Do NOT describe the paper beyond what the reviewer says. Do NOT add background facts, assumptions or inferences of your own. Do not judge whether a premise is true.
If the item is a question, write only what the question presupposes about the paper.
Use 1 to 4 premises.

Return ONLY JSON: {"premises": ["...", "..."], "conclusion": "..."}"""

# Natural-science reviews: each premise is typed paper / knowledge / opinion; only paper premises are checked.
DECOMP_SCIENCE = """You reconstruct ONE peer-review item as a short argument, so that each step can be fact-checked separately.

List the premises the reviewer asserts or directly presupposes, and give each premise a type:
- "paper": a claim about what the paper states, reports, shows, cites or omits.
- "knowledge": a factual claim about the world or about other work that does not depend on this paper (for example what a standard method does, or what another paper introduced).
- "opinion": an evaluation, norm or recommendation (for example "X is limited", "Y is unconvincing", "the paper should do Z", "a good paper would ...").

Rules:
- Each premise is ONE self-contained declarative sentence. Say "the paper" and name the method, dataset, table, figure, section or equation it is about.
- Make "paper" and "knowledge" premises concrete and checkable. Do not use vague or evaluative words in them (little, limited, clearly, severe, sufficient, convincing, well). Put such evaluations in "opinion" premises.
- For an omission, state exactly what is missing (for example "The paper does not report results on dataset D"), not how much is missing.
- Keep each premise as close to the reviewer's own claim as possible. Do not add facts the reviewer did not assert or imply. Do not describe the paper beyond what the reviewer says. Do not judge whether a premise is true.
- If the item is a question, write only what the question presupposes about the paper.
- Use 1 to 5 premises in total.

Return ONLY JSON: {"premises": [{"text": "...", "type": "paper"}, ...], "conclusion": "..."}"""

DECOMP_USER = '### Review item\n%s\n\nReturn ONLY the JSON object.'

# ------------------------------------------------------------------------------- LLM judge -----
# Holistic LLM baseline and LLM verifiers of paper-grounded premises.
LLM_JUDGE_SYSTEM = 'You judge whether statements are true according to a given evidence text from a scientific paper. Judge each statement only from the evidence text. For each statement, give the probability (integer 0-100) that it is true according to the evidence. Return ONLY JSON: {"p": {"<id>": <int>, ...}}.'

# Knowledge premises (LLM verifiers) and the routing question: general knowledge allowed.
NEUTRAL_SYSTEM = 'You answer yes/no questions about the given text. For each statement, give the probability (integer 0-100) that it is true, using the given text and general knowledge. Return ONLY JSON: {"p": {"<id>": <int>, ...}}.'

# ------------------------------------------------------------------- decision-model questions -----
ROUTE_Q = 'Whether this statement is true can be decided from the content of the paper it refers to alone, without outside knowledge.'
VERIFY_PAPER_Q = 'The proposition is true according to the paper.'
VERIFY_KNOWLEDGE_Q = 'The proposition is true.'
# Natural-science reviews: replaces VERIFY_PAPER_Q; the premise score is one minus its probability.
CONTRADICTION_Q = 'The paper contains information that contradicts the proposition.'
HOLISTIC_Q = "The review item's main criticism is factually correct about the paper."

# 16 phrasings of the verification question, asked in one call per premise. Phrasings 8-15 are inverted.
PAPER_PHRASINGS = [
    'The proposition is true according to the paper.',
    'The paper states or clearly implies the proposition.',
    'A careful reader of the paper would accept the proposition as an accurate description of the paper.',
    'The proposition accurately describes the paper.',
    'The paper supports the proposition.',
    'The proposition is consistent with what the paper reports.',
    'Someone who has read the paper closely would agree with the proposition.',
    'The proposition is a faithful statement about the content of the paper.',
    'The paper contains information that contradicts the proposition.',
    'The proposition misrepresents the paper.',
    'The proposition is false according to the paper.',
    'The proposition misreads or overlooks something that the paper states.',
    'The paper reports something that shows the proposition is wrong.',
    'A careful reader of the paper would reject the proposition.',
    'The proposition is inaccurate as a description of the paper.',
    'The proposition conflicts with the content of the paper.',
]
KNOWLEDGE_PHRASINGS = [
    'The proposition is true.',
    'The proposition is correct.',
    'The proposition is a true statement.',
    'An expert in the field would agree with the proposition.',
    'The proposition is accurate.',
    'The proposition is factually correct.',
    'The proposition reflects established knowledge in the field.',
    'The proposition is a well-supported claim.',
    'The proposition is false.',
    'The proposition is incorrect.',
    'An expert in the field would reject the proposition.',
    'The proposition contains a factual error.',
    'The proposition is inaccurate.',
    'The proposition is misleading.',
    'The proposition contradicts established knowledge in the field.',
    'The proposition is wrong.',
]
INVERTED = frozenset(range(8, 16))
