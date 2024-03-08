#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:46:22 2024

@author: deekshitadoli
"""
import langchain_community
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer

import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from llama_index import download_loader
from langchain.chat_models import ChatOpenAI
from llama_index import VectorStoreIndex, download_loader, LLMPredictor, ServiceContext

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI, openai

urls = [
    "https://hr.uw.edu/dso/",
    "https://hr.uw.edu/dso/services/",
    "https://hr.uw.edu/dso/services/services-for-faculty-and-staff/",
    "https://hr.uw.edu/dso/services/managersrole/",
    "https://hr.uw.edu/dso/services/uw-job-applicants/",
    "https://hr.uw.edu/dso/services/matriculated-students/",
    "https://hr.uw.edu/dso/services/services-for-students/",
    "https://hr.uw.edu/dso/services/services-for-the-public/",
    "https://hr.uw.edu/dso/deaf-or-hard-of-hearing/overview/",
    "https://hr.uw.edu/dso/deaf-or-hard-of-hearing/faculty-guide-for-zoom-classes-with-interpreters-captioners/",
    "https://hr.uw.edu/dso/deaf-or-hard-of-hearing/faculty-guide-zoom-small-group-or-1-to-1-interpreted-meetings/",
    "https://hr.uw.edu/dso/deaf-or-hard-of-hearing/interpreter-guide-for-zoom-classes/",
    "https://hr.uw.edu/dso/deaf-or-hard-of-hearing/student-guide-for-working-with-interpreters-in-zoom-classes/",
    "https://hr.uw.edu/dso/disability-parking/students/",
    "https://hr.uw.edu/dso/disability-parking/employees/",
    "https://hr.uw.edu/dso/disability-parking/visitors/",
    "https://hr.uw.edu/dso/service-animals/",
    "https://hr.uw.edu/dso/ergonomics/",
    "https://hr.uw.edu/dso/contacts/",
    "https://hr.uw.edu/dso/additional-resources/deaf-and-hard-of-hearing-service-providers/",
    "https://hr.uw.edu/dso/additional-resources/accommodation-event-notice/",
    "https://hr.uw.edu/dso/additional-resources/dso-flyer/",
    "https://hr.uw.edu/dso/additional-resources/resources-from-dso-partners/",
    "https://depts.washington.edu/uwdrs/",
    "https://depts.washington.edu/uwdrs/drs/",
    "https://depts.washington.edu/uwdrs/drs/who-we-are/",
    "https://depts.washington.edu/uwdrs/drs/meet-our-staff/",
    "https://depts.washington.edu/uwdrs/drs/student-testimonials/",
    "https://depts.washington.edu/uwdrs/drs/legislation/",
    "https://depts.washington.edu/uwdrs/drs/contact-us/",
    "https://depts.washington.edu/uwdrs/prospective-students/getting-started/",
    "https://depts.washington.edu/uwdrs/prospective-students/documentation-guidelines/",
    "https://depts.washington.edu/uwdrs/prospective-students/transition-resources/",
    "https://depts.washington.edu/uwdrs/current-students/student-rights-responsibilities/",
    "https://depts.washington.edu/uwdrs/current-students/accommodations/housing/",
    "https://depts.washington.edu/uwdrs/current-students/accessplanning-meeting/",
    "https://depts.washington.edu/uwdrs/current-students/mydrs-video-tutorials/",
    "https://depts.washington.edu/uwdrs/parents-and-family/",
    "https://depts.washington.edu/uwdrs/current-students/accommodations/",
    "https://depts.washington.edu/uwdrs/current-students/services-request-timeline/",
    "https://depts.washington.edu/uwdrs/current-students/mydrs-video-tutorials/",
    "https://depts.washington.edu/uwdrs/current-students/career-and-beyond/",
    "https://depts.washington.edu/uwdrs/current-students/reconsideration/",
    "https://depts.washington.edu/uwdrs/resources/",
    "https://depts.washington.edu/uwdrs/faculty/syllabus-statement/",
    "https://depts.washington.edu/uwdrs/faculty/supporting-students-with-disabilities/",
    "https://depts.washington.edu/uwdrs/faculty/faculty-responsibilities/",
    "https://depts.washington.edu/uwdrs/faculty/timeline/",
    "https://depts.washington.edu/uwdrs/faculty/faqs/",
    "https://depts.washington.edu/uwdrs/faculty/faculty-resources/",
    "https://depts.washington.edu/uwdrs/staff-resources/",
    "https://depts.washington.edu/uwdrs/faculty/accommodation-reconsideration/",
    "https://depts.washington.edu/uwdrs/current-students/accommodations/alternative-testing/",
    "https://depts.washington.edu/uwdrs/faculty/alternative-testing-services/",
    "https://depts.washington.edu/uwdrs/news-updates/",
    "https://facilities.uw.edu/form/ada-barrier?ref=www/admin/ada/barrier",
    "https://depts.washington.edu/uwdrs/current-students/mydrs-student-guide/",
    "https://depts.washington.edu/uwdrs/complaint-mediation/",
    "https://www.ehs.washington.edu/resource/evacuation-topics-individuals-disabilities-198",
    "https://www.ehs.washington.edu/system/files/resources/Focus-Sheet-disabilities.pdf",
    "https://depts.washington.edu/uwdrs/wp-content/uploads/2022/03/DRS-Healthcare-Provider-Form-2022-March-Fillable.pdf",
    "https://depts.washington.edu/uwdrs/wp-content/uploads/2016/07/Release-of-Information-Form-FILLABLE-FORM.pdf",
    "https://ihdd.org/",
    "https://uwctds.washington.edu/",
    "https://disabilitystudies.washington.edu/",
    "https://depts.washington.edu/ic/",
    "http://depts.washington.edu/uwautism/index.php",
    "https://depts.washington.edu/dcenter/",
    "https://depts.washington.edu/dcenter/alliance-for-disability-law-and-justice-adlj/",
    "https://depts.washington.edu/dcenter/asl-club/",
    "https://depts.washington.edu/dcenter/asuw-student-disability-commission/"

    ]

loader = AsyncChromiumLoader(urls) #gives html format files
#loader = SeleniumURLLoader(urls=urls)
#loader = UnstructuredURLLoader(urls=urls)

docs = loader.load()
print(type(docs[0])) #for AsyncChromiumLoader <class 'langchain_core.documents.base.Document'>

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
f= open("asy_html2text.txt","w+")
print(docs_transformed[0].metadata['source']) #
for doc in docs_transformed:
    f.write(doc.page_content)
    
    
text_splitter = CharacterTextSplitter(separator='\n', 
                                  chunk_size=1000, 
                                  chunk_overlap=200)

docs = text_splitter.split_documents(docs_transformed)

embeddings = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
#embeddings = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
vectorStore = FAISS.from_documents(docs, embeddings)

with open("asy_html2text_vectors.pkl", "wb") as f:
    pickle.dump(vectorStore, f)

with open("asy_html2text_vectors.pkl", "rb") as f:
    vectorStore = pickle.load(f)

CHAT_MODEL = 'gpt-3.5-turbo'
llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, api_key=st.secrets['OPENAI_API_KEY'])

retriever = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorStore.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True
)

query = "Who is dimitri? What is their full name and contact info?"

result = retriever({"question": query})

def format_escape_characters(s):
    return s.replace('"', '\\"').replace("\n", "\\n")


def transform_to_json(result):
    formatted_result_string = format_escape_characters(result["answer"]+result["sources"])
    return f"""
        {{
        "result": "{formatted_result_string}"
        }}"""

res = transform_to_json(result)
print(res)

'''
BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=urls)
print(type(documents[0]))
f= open("india_bsr.txt","w+")
for doc in documents:
    f.write(doc.text)
'''
    
