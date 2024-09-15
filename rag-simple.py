from getpass import getpass
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import configparser

# The token will expire on Tue, Oct 15 2024
config = configparser.ConfigParser()
config.read('./config.ini')
my_token = config['gitHub']['access_token']
ACCESS_TOKEN = getpass(my_token)


# -------------------------- Load documents --------------------------
# Load issues from https://github.com/huggingface/peft
loader = GitHubIssuesLoader(
    repo_path="huggingface/peft",
    access_token=ACCESS_TOKEN,
    include_prs=False,
    state='all'
)
docs = loader.load()

# -------------------------- Chunking --------------------------
# Chunking because embeded models we will me using may not be able to handle large documents as input
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)

# To create document chunk embeddings we’ll use the HuggingFaceEmbeddings and the BAAI/bge-base-en-v1.5
# (https://huggingface.co/BAAI/bge-base-en-v1.5) embeddings model.
# Embedding leaderboard: https://huggingface.co/spaces/mteb/leaderboard

# -------------------------- Create vector store --------------------------
# FAISS is a library for efficient similarity search
db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

# -------------------------- Retriever --------------------------
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# -------------------------- Load the LLM --------------------------
# Model of choice: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
my_model_name = 'HuggingFaceH4/zephyr-7b-beta'
bnb_config = BitsAndBytesConfig.load(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(my_model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(my_model_name)

# -------------------------- Setup the chain --------------------------
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()


# -------------------------- Run the chain --------------------------
retriever = db.as_retriever()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
while True:
    question = input("Question: ")
    if question == "exit":
        break
    result = llm_chain.invoke({'context': '', 'question': question})
    print(result)