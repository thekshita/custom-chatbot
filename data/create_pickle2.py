#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:04:06 2024

@author: deekshitadoli
"""

import requests
from bs4 import BeautifulSoup
import faiss
import pickle
import numpy as np 
# Function to extract text from a webpage
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        return text
    except Exception as e:
        print(f"Error fetching text from {url}: {e}")
        return None

# List of URLs
urls_list = [
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

# Extract text from each URL
corpus = []
for url in urls_list:
    text = extract_text_from_url(url)
    if text:
        corpus.append(text)

# Tokenize and create vectors using FAISS
d = 300  # Dimension of the vectors
index = faiss.IndexFlatL2(d)  # L2 distance for similarity search
vecs = []
for text in corpus:
    # Convert text to vectors (you need to implement your vectorization method here)
    # Example: using TF-IDF, Word2Vec, or any other method
    # Here, I'm just using a placeholder vector of zeros
    vec = [0] * d
    vecs.append(vec)

# Convert list of vectors to numpy array
vecs_np = np.array(vecs, dtype=np.float32)

# Add vectors to the FAISS index
index.add(vecs_np)

# Save the FAISS index to a pickle file
with open('faiss_index.pickle', 'wb') as f:
    pickle.dump(index, f)

# Save the corpus (list of texts) to a pickle file
with open('corpus.pickle', 'wb') as f:
    pickle.dump(corpus, f)

print("Vector store created and saved successfully!")
