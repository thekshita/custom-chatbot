# Equity Engine for Inclusive Learning

Equity Engine is a context-based conversational memory chatbot aimed at facilitating access to the Disability Resources for Students (DRS) at the University of Washington (UW), addressing the growing need for improved access to information as the student population expands.

## Tools Used

**Chat Model:**

**Embedding Model:**


**Document loader and transformer:**

**Vector store:**

**Retriever:**

**Streamlit Interface:**
The chatbot interface is built using Streamlit, allowing users to interact intuitively with the bot through a web-based interface.

## System Architecture
![Equity Engine System Architecture](https://drive.google.com/drive/u/0/folders/1iooncDLwsY3s_SCQEv11F32dCsQAspN7)
main.py
query_service.py
agent.py
data/create_pickle.py
tools/QnA.py
utils.py


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
