from app.services.rag import ask_rag

question = "Does Faizan know Python?"

answer = ask_rag(question)

print("\nAnswer:")
print(answer)