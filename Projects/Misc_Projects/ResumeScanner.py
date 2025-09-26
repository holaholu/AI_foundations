#importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report

#Sample resumes and job description data
data = {
    'resume_id' :[1,2,3],
    'resume_text' :[
        "Experienced software engineer with 5 years of experience in developing scalable and secure web applications.",
        "Software developer with expertise in Java,cloud computing, and database management.",
        "Data analyst with a strong background in statistical analysis and data visualization"
        ]}
job_description = " Looking for a data scientist skilled inPython, machine learning, and data visualization"

#Convert the data into a pandas dataframe
df = pd.DataFrame(data)
#print("Resumes:",df)

#Combine the job description with each resume for TF-IDF vectorization
documents = df['resume_text'].tolist()
documents.append(job_description)
print("Documents:",documents)

#Initialize the TF-IDF vectorizer and fit the documents
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

#Calculate the cosine similarity between the job description and each resume
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten() #Calculate the cosine similarity between the job description(last element) and each resume ( all elements except the last one)

#Display similarity scores for each resume
df['similarity_score'] = similarity_scores
print("\nResume Similarity Scores:\n",df[['resume_id','similarity_score']])

#Identify resumes that match the job requirements ( threshold can be adjusted)
threshold = 0.2
matching_resumes = df[df['similarity_score'] >= threshold]
print("\nMatching Resumes:\n",matching_resumes[['resume_id','similarity_score']])