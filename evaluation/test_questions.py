TEST_QUESTIONS = [

    # Umair.pdf tests
    {
        "question": "Who is Umair?",
        "expected_source": "Umair.pdf",
    },
    {
        "question": "What company is Umair associated with?",
        "expected_source": "Umair.pdf",
    },
    {
        "question": "What is Umair's role?",
        "expected_source": "Umair.pdf",
    },
    {
        "question": "What information is mentioned about Umair?",
        "expected_source": "Umair.pdf",
    },

    # Fayard Law.pdf tests
    {
        "question": "What is Fayard Law's address?",
        "expected_source": "Fayard Law.pdf",
    },
    {
        "question": "What services does Fayard Law provide?",
        "expected_source": "Fayard Law.pdf",
    },

    # Negative / unrelated tests
    {
        "question": "What is the weather today?",
        "expected_source": None,
    },
    {
        "question": "What information is available about a person who is not in the documents?",
        "expected_source": None,
    },
    {
        "question": "What is the capital of France?",
        "expected_source": None,
    },
]