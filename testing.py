# import os
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain_huggingface import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables.base import RunnableSequence

# # Load environment variables from .env file
# load_dotenv()

# # Get the API token from environment variables
# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Check if the API token is loaded correctly
# if not api_token:
#     raise ValueError("The Hugging Face API token is not set. Please check your .env file.")

# # Load the tokenizer and model using transformers
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", token=api_token)
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", token=api_token)

# # Create a text generation pipeline
# pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=250)

# # Initialize LangChain with the HuggingFace pipeline
# llm = HuggingFacePipeline(pipeline=pipe)

# # Define a prompt template
# prompt_template = PromptTemplate(template="What is the full form of {query}?", input_variables=["query"])

# # Create a RunnableSequence with the prompt template and HuggingFace pipeline
# sequence = RunnableSequence(prompt_template, llm)

# # Example usage
# input_text = {"query": "MS Word"}
# response = sequence.invoke(input_text)
# print("Response:", response)




# import requests

# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
# headers = {"Authorization": f"Bearer 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"}

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

# data = query({
#     "inputs": "Hello my name is zain",
#     "parameters": {
#         "max_new_tokens": 250,
#         "temperature": 1,
#         "top_p": 0.9
#     },
#     "options": {
#         "use_cache": True,
#         "wait_for_model": True
#     }
# })

# print(data)




# import requests

# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
# headers = {"Authorization": f"Bearer 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"}

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     try:
#         response.raise_for_status()  # Check for HTTP errors
#         return response.json()
#     except requests.exceptions.HTTPError as err:
#         raise SystemExit(err)

# data = query({
#     "inputs": "____________________________",
#     "parameters": {
#         "max_new_tokens": 250,
#         "temperature": 1,
#         "top_p": 0.9
#     },
#     "options": {
#         "use_cache": True,
#         "wait_for_model": True
#     }
# })

# # Print the structured response
# if 'error' in data:
#     print(f"Error: {data['error']}")
# else:
#     for idx, generated in enumerate(data):
#         print(f"Generated text {idx+1}: {generated['generated_text']}")
