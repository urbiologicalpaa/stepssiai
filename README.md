# Textbook Question Answering System

This project demonstrates a Textbook Question Answering System that uses a combination of FAISS for vector-based retrieval and a language model for generating answers. The system is designed to efficiently retrieve and answer questions based on the content of selected textbooks.

## Table of Contents

1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Dependencies](#dependencies)
4. [How to Run](#how-to-run)
5. [Selected Textbooks](#selected-textbooks)

## Overview

The Textbook Question Answering System performs the following tasks:
- Embeds and stores textbook content using FAISS for efficient retrieval.
- Uses a SentenceTransformer model to embed queries.
- Retrieves relevant passages using FAISS.
- Uses a language model (RoBERTa) to generate answers based on the retrieved passages.
- Provides a user interface for inputting queries and viewing answers using Streamlit.

## Setup Instructions

1. **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Required Packages:**

    ```bash
    pip install streamlit faiss-cpu transformers sentence-transformers pyngrok
    ```

3. **Obtain an ngrok Authtoken:**

    - Sign up for an ngrok account [here](https://dashboard.ngrok.com/signup).
    - After signing up, get your authtoken from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

4. **Set Up the ngrok Authtoken:**

    Replace `"YOUR_NGROK_AUTH_TOKEN"` in the provided script with your actual ngrok authtoken.

## Dependencies

- **Streamlit:** For creating the user interface.
- **FAISS:** For efficient similarity search and clustering of dense vectors.
- **Transformers:** For using pre-trained language models.
- **Sentence-Transformers:** For generating embeddings of the query and texts.
- **Pyngrok:** For creating a public URL to access the Streamlit app.

## How to Run

1. **Prepare the Textbook Data:**

    - Ensure your textbook content is preprocessed and stored in a format compatible with FAISS and the SentenceTransformer model.

2. **Run the Main Script:**

    Use the following Python script to set up and run the Streamlit app:

    ```python
    import streamlit as st
    import faiss
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
    from sentence_transformers import SentenceTransformer
    from pyngrok import ngrok
    import os

    # Save the main code to a file
    with open('app.py', 'w') as f:
        f.write("""
    import streamlit as st
    import faiss
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
    from sentence_transformers import SentenceTransformer

    # Load FAISS index
    index = faiss.read_index("textbook_index.faiss")

    # Load the summarization model and tokenizer
    sbert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_model = SentenceTransformer(sbert_model_name)
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    # Initialize the question answering pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Dummy summarized texts and metadata for demonstration
    summarized_texts = [
        "Computational biology is the study of biology using computational techniques.",
        "Genetic information is stored in the DNA of living organisms.",
        "Proteins are made of amino acids and perform various functions in the body.",
        # Add more dummy summarized texts corresponding to your data
    ]
    metadata = [
        {"title": "Textbook 1", "page_number": 10},
        {"title": "Textbook 2", "page_number": 23},
        {"title": "Textbook 3", "page_number": 45},
        # Add more metadata corresponding to your data
    ]

    # Example retrieval function using FAISS
    def retrieve_faiss(query, index, k=5):
        query_embedding = sbert_model.encode(query, convert_to_tensor=True)  # Get the Tensor
        query_embedding = query_embedding.detach().numpy().astype('float32')  # Convert to NumPy and then to float32
        query_embedding = query_embedding.reshape(1, -1)  # Reshape the query embedding to a 2D array
        D, I = index.search(query_embedding, k)
        results = [(summarized_texts[i], metadata[i], D[0][i]) for i in I[0]]
        return results

    # Streamlit User Interface
    st.title("Textbook Question Answering System")

    query = st.text_input("Enter your query:", "")

    if query:
        # Retrieve relevant passages using FAISS
        results = retrieve_faiss(query, index)
        
        st.write(f"Top {len(results)} relevant passages retrieved:")

        for i, (text, meta, score) in enumerate(results):
            st.write(f"**Passage {i+1}:**")
            st.write(f"Text: {text}")
            st.write(f"Title: {meta['title']}, Page Number: {meta['page_number']}")
            st.write(f"Relevance Score: {score:.4f}")

            # Use the QA pipeline to generate an answer
            answer = qa_pipeline(question=query, context=text)
            st.write(f"**Answer:** {answer['answer']}\n")
    """)

    # Set the ngrok authtoken
    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

    # Run the Streamlit app
    os.system('streamlit run app.py &')

    # Create a public URL for the Streamlit app
    public_url = ngrok.connect(port='8501')
    print(f"Streamlit app is live at: {public_url}")
    ```

3. **Access the Streamlit App:**

    - Once you run the script, it will output a public URL. Open this URL in your browser to access the Streamlit app.
    - You can input queries into the interface and view the retrieved passages and answers.

## Selected Textbooks

Below are the titles and links to the selected textbooks used for content extraction:

1. **Title:** Bailey & Scott's Diagnostic Microbiology 14e
   - **Link:** [Computational Biology](https://drive.google.com/file/d/1oKzZCe1TUmWxUHeeg7vSEv5ecz8Gg0Hu/view?usp=drive_link)

2. **Title:** BIOINFORMATICS AN INTRODUCTION_BY_J.RAMSEDEN
   - **Link:** [Introduction to Bioinformatics](https://drive.google.com/file/d/11PdPYTtX3YiXfte7xee8LYr0uKdL-P09/view?usp=sharing)

3. **Title:** bioethics_and_biosafety_in_biotechnology
   - **Link:** [Principles of Genetics](https://drive.google.com/file/d/10_jK8llOBLvlW-7lregDHPF2Dog8pHOf/view?usp=drive_link)

*(Note: The textbook titles and links are examples. Replace them with the actual titles and links of the textbooks used in your project.)*

---

Feel free to modify the README file to fit the specifics of your project, including the actual titles and links to the textbooks you used.
