from app.rag.qa import answer_codebase_question

question = "Where does the application connect to Ollama?"

answer = answer_codebase_question(question)

print("\nCodeSage Answer:\n")
print(answer)