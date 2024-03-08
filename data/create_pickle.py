#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 00:08:55 2024

@author: deekshitadoli
"""

#from dotenv import load_dotenv


import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from llama_index import download_loader
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer

import os
import streamlit as st

import bs4


def create_pkl(category: str, urls: list):
    loaders = AsyncChromiumLoader(urls)
    #loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(data)

    
    text_splitter = CharacterTextSplitter(separator='\n', 
                                      chunk_size=1000, 
                                      chunk_overlap=200)

    docs = text_splitter.split_documents(docs_transformed)

    embeddings = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)
    
    with open(f"{category}_vectors.pkl", "wb") as f:
        pickle.dump(vectorStore_openAI, f)
    
    with open(f"{category}_vectors.pkl", "rb") as f:
        vectorStore = pickle.load(f)
        
        
    
    
        
        
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


create_pkl("drs", urls)
           
urls = [
       "https://www.washington.edu/studentlife/living-dining/",
       "https://hfs.uw.edu/Live/Undergraduate-Communities",
       "https://hfs.uw.edu/live-on-campus/graduate-student-apartments",
       "https://www.ielp.uw.edu/life-at-uw/housing/homestays",
       "https://www.ielp.uw.edu/life-at-uw/housing/temporary-housing",
       "https://uwifc.com/#!housing/c1lte",
       "https://uwpanhellenic.com/",
       "https://hfs.uw.edu/Eat/Resident-Dining",
       "https://hfs.uw.edu/Eat/Residence-Hall-dining-plan",
       "https://hfs.uw.edu/Eat/Apartment-dining-plan",
       "https://hfs.uw.edu/Experience/Student-Jobs"
   ]
create_pkl("hfs", urls)
