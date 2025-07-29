from langchain.tools import Tool


def create_refusal_tool():
    return Tool(
        name="Refusal",
        func=lambda q: (
            "Sorry, I can only answer questions about the healthcare dataset, not about programming, code, unrelated topics, or medical advice. "
            "For any medical concerns, please consult a licensed healthcare professional."
        ),
        description="Use for any question outside the scope of the healthcare dataset, programming/code/meta-requests, or medical advice."
    )
