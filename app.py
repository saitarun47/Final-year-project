import streamlit as st
from rag_system import RAGSystem

# Initialize the RAG system
rag_system = RAGSystem()

# Set up the Streamlit app layout
st.title("Pedagogy Suggestion System")
st.write("Enter a course name to receive effective pedagogy suggestions.")

# Input field for course name
course_name = st.text_input("Course Name:")

if st.button("Get Suggestions"):
    if course_name:
        with st.spinner("Generating suggestions..."):
            try:
                # Generate suggestions based on the course name
                suggestions = rag_system.generate_suggestion(course_name)
                st.subheader("Suggested Pedagogies:")
                st.write(suggestions)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a course name.")

# Add footer information
st.write("Developed by Sai Tarun")
