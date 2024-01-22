import os
import pprint
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Union
from loguru import logger as log

import tiktoken
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import Document
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from pydantic import BaseModel, Field
import streamlit as st
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def _calc_tokens(splits: List[Document]) -> int:
    tokens = 0

    for doc in splits:
        encoding = tiktoken.get_encoding('cl100k_base')
        tokens += len(encoding.encode(doc.page_content))

    return tokens


class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class Assistant:
    def __init__(self):
        load_dotenv()
        self.db_dir = 'docs/chroma/'
        self.embedding = AzureOpenAIEmbeddings(azure_deployment="ada_dev")
        self.llm = AzureChatOpenAI(
            azure_deployment="35_turbo",
            model_name="gpt-35-turbo",
            temperature=0
        )

        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")
        os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv(
            "AZURE_OPENAI_ENDPOINT")
        self.make_template()

    def run(self):
        st.title('Гаррі Поттер асистент')

        instruction = st.text_input('Питання', '')

        if st.button('Згенерувати відповідь'):
            result, docs = self.stuff_search(instruction)
            st.subheader('Відповідь')
            st.text(result)
            st.header('Знайдені чанки')
            for doc in docs:
                st.subheader(f'Сторінка {doc.metadata.get("page")}')
                st.text(doc.page_content)

    def make_template(self):
        template = """Ти ШІ консультант. Твоя задача відповідати на запитання користувачів. Запитання будуть про книгу "Гаррі Поттер та філософський камінь". Додатково тобі будуть надані частини тексту з книги в якості контексту, з яких ти повинен надати відповідь. Ти повинен використовувати для відповіді лише наданий контекст і не додумувати нічого від себе. Якщо в частинах тексту немає відповідної інформації, щоб надати відповідь - вибачся та скажи, що не знаєш відповіді. ВАЖЛИВО відповідати виключно УКРАЇНСЬКОЮ мовою.
        Контекст:
        {context}
        Запитання: {question}
        Відповідь:"""
        self.prompt = PromptTemplate.from_template(template)

    def load_pdf(self, file_name: str) -> List[Document]:
        log.info("Loading pdf")
        loader = PyPDFLoader(f"files_to_load/{file_name}")
        return loader.load()

    def split_documents(self, pages: List[dict]) -> Union[List[Document], None]:
        log.info("Splitting pdf")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        return text_splitter.split_documents(pages)

    def save_in_db(self, splits: List[Document]):
        log.info("Saving chunks in db")
        if len(splits) == 0:
            log.warning(
                "There are no splits to save in db. Please provide them in arguments or call the split_documents(headers_to_split, pages) method")
            return None

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding,
            persist_directory=self.db_dir
        )

        log.info(f"{vectordb._collection.count()} rows were saved")
        log.info(f"{_calc_tokens(splits)} tokens were affected")
        return True

    def stuff_search(self, question: str):
        vectordb = Chroma(persist_directory=self.db_dir,
                          embedding_function=self.embedding)

        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

        result = qa_chain({"query": question})
        log.info(f'Questing: {question}')
        log.info(f'Result: {result["result"]}')
        log.info("DOCUMENTS:")
        for doc in result["source_documents"]:
            log.info(doc)

        return result["result"], result["source_documents"]

if __name__ == "__main__":
    assistant = Assistant()
    vectordb = Chroma(persist_directory="docs/chroma/",
                    embedding_function=assistant.embedding)
    if(len(vectordb.get().get("documents")) == 0):
        pdf = assistant.load_pdf("Harry_Potter.pdf")
        splits = assistant.split_documents(pdf)
        assistant.save_in_db(splits)
    assistant.run()