import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

doc_name="Chronic diseases.xlsx"
def get_document(doc_name):
    df = pd.read_excel("Chronic diseases.xlsx")
    return df

doc=get_document(doc_name=doc_name)

def create_data(extracted_data):
    documents=[]
    for _, row in extracted_data.iterrows():
        text=f"""
        Chronic_Category={row['Chronic Disease Category']}
        Specialist={row['Specialist']}
        Chronic_Type={row['Types of chronic disease']}
        Primary_Symptoms={row['Primary Symptoms']}
        Secondary_Symptoms={row['Secondary Symptoms']}
        Hidden_Silent_signs={row['Hidden/Silent signs']}
        Severity={row['Severity']}
        Risk_factor={row['Risk factor']}
        Duration={row['Duration']}
        Age_Group={row['Age Group']}
        Gender={row['Gender']}
        Screening_questions={row['Screening questions']}
        Chatbot_Training={row['Chatbot Training questions']}
        Overlapping_Symptoms={row['Overlapping Symptoms']}
        Overlapping_Questions={row['Overlapping Questions']}
        """
        documents.append(Document(page_content=text))
        break
    return documents

document=create_data(extracted_data=doc)
print(document)

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model=get_embedding_model()

DB_FAISS_PATH="vectorstore/db_faiss"
db = FAISS.from_documents(
    document,
    embedding_model
)
db.save_local(DB_FAISS_PATH)