import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Load your dataset
df = pd.read_csv('data/pedagogy.csv')

# Clean and preprocess data
df = df.dropna()  # Remove rows with missing values
df.columns = df.columns.str.strip()

df['combined_text'] = (
    df['Course Name'].astype(str) + ' ' +
    df['Pedagogies used'].astype(str) + ' ' +
    df['Average Student Marks'].astype(str) + ' ' +
    df['Student Feedback'].astype(str)
)

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['combined_text'].tolist())

# Initialize Pinecone with the updated syntax
pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Define your index parameters
index_name = 'pedagogy-suggestion'
dimension = 384  # Dimension of 'all-MiniLM-L6-v2'

# Check if index exists; if not, create it
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',  # or your preferred metric
        spec=pinecone.ServerlessSpec(
            cloud='aws',  # or 'aws' if that's your preference
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Upsert vectors to Pinecone
for i, row in df.iterrows():
    index.upsert([(str(i), embeddings[i].tolist(), {
        'Course Name': row['Course Name'],
        'Pedagogies used': row['Pedagogies used'],
        'Average Student Marks': row['Average Student Marks'],
        'Student Feedback': row['Student Feedback']
        
    })])

print("Data preparation and indexing complete!")