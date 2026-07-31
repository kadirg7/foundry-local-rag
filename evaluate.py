import rag

# Test set: questions the report CAN answer + questions it CANNOT
TESTS = [
    # (question, should_answer)
    ("What is precooling?", True),
    ("How much food is lost due to lack of refrigeration?", True),
    ("What share of global emissions comes from the cold chain?", True),
    ("Why do developing countries lose more food?", True),
    ("What is the difference between precooling and cold storage?", True),
    ("How does refrigeration relate to food safety?", True),
    ("Who won the World Cup in 2018?", False),
    ("What is the capital of France?", False),
    ("How do I bake sourdough bread?", False),
    ("What is the price of Bitcoin?", False),
]

REFUSAL_MARKER = "i don't have that information"

print("Loading models...")
rag.init()
print("Running evaluation...\n")

correct = 0
for question, should_answer in TESTS:
    answer, results = rag.answer_query(question, top_k=3)
    top_score = results[0][1] if results else 0.0

    refused = REFUSAL_MARKER in answer.lower()
    answered = not refused

    ok = (answered == should_answer)
    correct += ok

    status = "PASS" if ok else "FAIL"
    kind = "answerable" if should_answer else "unanswerable"
    print(f"[{status}] ({kind}, top={top_score:.3f}) {question}")
    print(f"        -> {answer[:90]}\n")

print(f"Score: {correct}/{len(TESTS)} correct behavior")