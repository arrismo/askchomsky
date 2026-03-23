from main import get_query_engine


if __name__ == "__main__":
    query_engine = get_query_engine(similarity_top_k=5)
    question = "How does Chomsky’s “manufacturing consent” model explain modern social media platforms?"
    response = query_engine.query(question)
    print("\nQ:", question)
    print("\nA:", response)
