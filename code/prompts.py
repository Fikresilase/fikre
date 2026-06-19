"""Prompt construction — pure retrieval + system prompting (no copy-first, no HyDE)."""
import config as C

SYSTEM_TMPL = (
    "You are a pure answerer agent not assistant not expert. Answer ONLY in {language}.\n"
    "Below are example question-answer pairs in {language}. Match their STYLE, LENGTH, and "
    "after finding the answer the maches the question u should answer the question in the same style and wording and length" \
    "very important always have same word length with the answer dont miss any of the keywords mentioned there "
    "and keep the same lexical ordering as the examples. "
    "never ever add any useless words that are not in the answer or anything that dont serve purpose to the direct answer of the question. "
    "question; if none match, answer concisely from general knowledge.\n"
    "Do NOT add greetings, disclaimers, or markdown. Output only the answer."
)


def build_messages(question: str, exemplar_pairs, subset: str):
    """Return chat messages [system, user] for one row.

    exemplar_pairs: list of (train_question, train_answer).
    """
    language = C.SUBSET_LANG[subset]
    system = SYSTEM_TMPL.format(language=language)
    ex_block = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in exemplar_pairs)
    user = f"{ex_block}\n\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
