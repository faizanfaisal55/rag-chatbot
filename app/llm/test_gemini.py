from app.llm.gemini import generate_answer

question = "What skills does Faizan have?"

context = """
Faizan knows Python, Machine Learning, Deep Learning,
TensorFlow, PyTorch and OpenCV.
"""

answer = generate_answer(question, context)

print(answer)