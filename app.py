import streamlit as st
from myQaAssistant import qa

def main():
    st.title("My QA Assistant")
    query = st.text_input("Enter your query ")
    prompt = f"""
    <|system|>>
    You are an AI Assistant that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in context
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
    if query:
        st.write("Response")
        response = qa(prompt)['result']
        helpful_answer = response.split("Helpful Answer:")[1].strip()
        st.write(qa(prompt))
if __name__== '__main__':
    main()