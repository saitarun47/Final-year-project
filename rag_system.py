import os
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Client  # Ensure you have this import for the Groq client

load_dotenv()

class RAGSystem:
    def __init__(self):
        # Initialize Groq client
        self.client = Client(api_key=os.getenv('GROQ_API_KEY'))
        
        # Create an instance of Pinecone
        self.pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Initialize index
        self.index_name = 'pedagogy-suggestion'
        self.index = self.pc.Index(self.index_name)  # Use the new syntax for indexing
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_suggestion(self, course_name):
        # Create query embedding
        query = f"{course_name} "
        query_embedding = self.model.encode(query).tolist()

        # Query Pinecone using keyword arguments
        results = self.index.query(vector=query_embedding, top_k=5, include_metadata=True)

        # Prepare context for Groq
        context = ""
        for match in results['matches']:
            context += f"Course: {match['metadata']['Course Name']}\n"
            context += f"Pedagogies: {match['metadata']['Pedagogies used']}\n"
            context += f"Student Feedback: {match['metadata']['Student Feedback']}\n"
            context += f"Marks: {match['metadata']['Average Student Marks']}\n\n"

        # Generate suggestion using Groq
        prompt = f"""Based on the following information about similar courses and topics:

{context}

Suggest effective top 3 pedagogies for teaching the course "{course_name}". 
Provide a detailed explanation of why these pedagogies would be effective, considering the average student feedback and  average marks from similar courses.
"""

        # Call the Groq API to generate text
        chat_completion = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            model="mixtral-8x7b-32768",  # You can change this to the model you prefer
            temperature=0.7,
            max_tokens=2000,
        )

        # Extract and return the generated text
        return chat_completion.choices[0].message.content
