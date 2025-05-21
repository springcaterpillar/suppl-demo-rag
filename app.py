import os
import time
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from typing import List
import faiss
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Load API keys
#load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['USER_AGENT'] = 'myagent'

st.title("⚖️ Suppl - FDA Compliance Checker")
st.write("Analyze your marketing claim for FDA regulatory risk and explore supporting evidence from official warning letters.")

user_claim = st.text_area("📝 **Enter your marketing claim or statement:**", height=150)

if st.button("🚀 Analyze Claim"):
    with st.status("⚙️ Preparing analysis...") as status:
        st.write("🔄 Loading regulatory data...")
        data = pd.read_csv('fda_dietary_supplement_warning_letters_with_text.csv')
        fda_letters = data['Letter Text'].tolist()[:200]
        urls = data['URL'].tolist()
        status.update(label="✅ Data loaded successfully.")

        docs_list = [Document(page_content=letter, metadata={"source": urls[i]}) for i, letter in enumerate(fda_letters)]

        st.write("🔄 Splitting and embedding documents...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        embedding_model = OpenAIEmbeddings()
        index = faiss.IndexFlatL2(1536)
        vectorstore = FAISS(embedding_function=embedding_model, index=index, index_to_docstore_id={}, docstore=InMemoryDocstore())
        uuids = [str(uuid4()) for _ in range(len(doc_splits))]
        vectorstore.add_documents(doc_splits, ids=uuids)
        status.update(label="✅ Documents processed and embedded.")

        st.write("🔍 Retrieving documents related to your claim...")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
        search_query = f"Is this claim likely to trigger FDA enforcement? {user_claim}"
        retrieved_docs = retriever.invoke(search_query)
        status.update(label="✅ Retrieved top relevant documents.")

        st.write("🔍 Scoring document relevance...")
        class GradeDocuments(BaseModel):
            binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
 
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Document:\n{document}\n\nQuestion:\n{question}")
        ])
        relevance_chain = grade_prompt | llm.with_structured_output(GradeDocuments)

        graded_docs = []
        for doc in retrieved_docs:
            result = relevance_chain.invoke({"document": doc.page_content, "question": search_query})
            graded_docs.append({"doc": doc, "is_relevant": result.binary_score.lower() == 'yes'})

        status.update(label="✅ Documents graded for relevance.")

        st.write("📝 Generating regulatory risk assessment...")
        def format_docs(docs):
            return "\n".join(f"<doc{i+1}>:\nContent: {doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate([d['doc'] for d in graded_docs]))

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a regulatory risk assessment based on the provided documents. Be specific and concise. Also rewrite the original claim to follow the FDA regulations. Add disclaimers and warnings that are required by the FDA."),
            ("human", "Documents:\n{documents}\n\nQuestion:\n{question}")
        ])
        answer_chain = answer_prompt | llm | StrOutputParser()
        assessment = answer_chain.invoke({"documents": format_docs(graded_docs), "question": search_query})

        st.write("🔎 Extracting supporting snippets and legal references...")
        class HighlightWithLaws(BaseModel):
            segment: List[str]
            laws: List[List[str]]
            source: List[str]

        combined_parser = PydanticOutputParser(pydantic_object=HighlightWithLaws)
        combined_prompt_template = """You are an advanced assistant for document search and retrieval. You are provided with the following:
        1. A question.
        2. A generated answer based on the question.
        3. A set of documents that were referenced in generating the answer.

        Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to
        generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text
        in the provided documents.

        Ensure that:
        - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
        - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
        - (Important) If you didn't used the specific document don't mention it.

        Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

        <format_instruction>
        {format_instructions}
        </format_instruction>
        """
        combined_prompt = PromptTemplate(
            template=combined_prompt_template,
            input_variables=["documents", "question", "generation"],
            partial_variables={"format_instructions": combined_parser.get_format_instructions()}
        )
        combined_chain = combined_prompt | ChatGroq(model="llama-3.3-70b-versatile", temperature=0) | combined_parser
        combined_response = combined_chain.invoke({"documents": format_docs(graded_docs), "question": search_query, "generation": assessment})

        status.update(label="✅ Analysis complete.", state="complete")

        # Display Assessment
        st.markdown("## ✅ **Regulatory Risk Assessment**")
        st.write(assessment)

        # Display All Retrieved Documents with Relevance and Highlighted Laws
        st.markdown("## 📄 **All Retrieved Documents with Cited Laws**")
        if combined_response.segment:
            for idx, snippet in enumerate(combined_response.segment):
                st.markdown(f"**Document {idx + 1}:** {snippet}")
                source_url = combined_response.source[idx] if idx < len(combined_response.source) else "Unknown"
                st.markdown(f"[🔗 View Full Letter]({source_url})\n")
        else:
            st.write("No specific snippets could be identified.")
