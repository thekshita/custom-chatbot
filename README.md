
# Equity Engine for Inclusive Learning

Equity Engine is a context-based conversational memory chatbot that uses the state-of-the-art Retrieval Augmented Generation (RAG) framework to facilitate access to the information about the Disability Resources for Students at the University of Washington, Seattle.

## Features

**Retrieval-Augmented Generation (RAG):** The chatbot utilizes the RAG framework to combine retrieval-based methods with language generation, enabling it to provide contextually relevant responses sourced from university websites.

**LangChain Integration:** The chatbot leverages LangChain's document loaders, text splitters, embedding models, vector stores, and retrievers to process and retrieve information effectively.

**User-Friendly Interface:** Built with Streamlit, the chatbot offers a seamless interaction experience, allowing users to ask questions and receive instant responses.

**Continuous Improvement:** The chatbot includes mechanisms for feedback collection and iterative refinement to enhance accuracy and user satisfaction over time.

## System Architecture
![Equity Engine - Draft Poster pptx](https://github.com/sealroboticsuw/equity-engine/blob/main/sys_arch.png)

## Methodology

The development of this chatbot follows a structured methodology outlined in the accompanying documentation:

**Data Collection:** URLs representing various university website sections are compiled to serve as primary information sources.

**Data Processing:** LangChain's document loaders and text splitters are used to preprocess HTML content and extract relevant text chunks.

**Embedding Generation:** OpenAIEmbeddings are employed to generate semantic representations of text data for efficient retrieval.

**Vector Storage:** FAISS is utilized to create a vector store for storing and querying computed embeddings.

**Retrieval-Based Chatbot:** The chatbot is developed using LangChain's RetrievalQAWithSourcesChain to retrieve answers to user queries based on similarity of embeddings.

**User Interface:** Streamlit is employed to design a user-friendly interface, enabling seamless interaction with the chatbot.

## Code Overview 
**1. main.py:**
Manages the Streamlit application interface for the university chatbot.
Handles user interaction, loads/saves chat history, and queries the AI model for responses.

**2.query_service.py:**
Handles user queries by either utilizing a general AI model or specialized agents based on the user's choice.
Manages responses from both general AI models and specialized agents.

**3. agent.py:**
Manages interactions with the conversational AI model, executing queries, and handling responses.
Utilizes tools such as QnA for specific domains to enhance responses.

**4. utils.py:**
Provides utility functions for formatting responses, checking response formats, and data transformation.

**5. tools/QnA.py:**
Creates a vector database tool for specialized queries using RetrievalQAWithSourcesChain.
Facilitates retrieving answers from pre-built vector databases, useful for handling specific domain queries.

**6. data/create_pickle.py:**
Generates pickle files containing vector representations of text data from university URLs.
Utilizes web scraping and text processing to gather data for vectorization, aiding in specialized query handling.

## Getting Started
To run the chatbot locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/your-username/university-chatbot.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run main.py
```
4. Interact with the chatbot through the web interface.

## Usage
1. Upon running the Streamlit app, user can select a query category based on the department they wish to enquire about or just go with the 'General' option. 
2. Users can then input their queries into the chat interface. 
3. The chatbot will process the query, generate a response using the OpenAI model, and display it in the chat interface.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments
We would like to thank our project sponsor Dr. Sep Makhsous and our capstone instructor Dr. Megan Hazen for their invaluable support and guidance throughout the project.  
