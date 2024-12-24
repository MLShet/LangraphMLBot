import json
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from states.state import OverallState


PROMPT = """
    You are a skilled data analyst specializing in domain identification from datasets. 
    Your task is to analyze the provided information to determine the dataset's domain (e.g., finance, health, education, industry). 

    You are given:
    1. **Human Expert Description**: "{description}" - This is a textual summary from a domain expert.
    2. **Dataset Metadata**: The dataset's structure is described using its **columns** : "{columns}" and **index** : "{index}".

    ### Your Tasks:
    - Carefully analyze the human expert description to understand the dataset's domain.
    - If the description is unclear or ambiguous, examine the dataset's metadata (columns and index names) to infer the domain.
    - If the index or columns suggest unique identifiers, categories, or domain-specific terminology, use this information to refine your inference.
    - If you are unable to confidently identify the domain, return an empty string.

    ### Output Format:
    Provide your response in the following JSON format:
    {format_instructions}"""

def domain_identification(state: OverallState):
    """
    This function classifies the domain of the dataset and provides a reason for the domain choice.
    The classification is based on the presence of domain-specific keywords, unique identifiers, 
    or terminology within the dataset.
    
    Examples of domains include finance, healthcare, education, industry, etc.
    """
    class OutputModel(BaseModel):
        """ The output model for the domain classification."""
        type: str = Field(..., description="""The domain of the dataset. 
                            For example, finance, health, education, industry etc.""")
        cause: str = Field(..., description="""The domain choice is based on the presence of specific keywords,
                            unique identifiers, or domain-specific terminology within the dataset,
                            which align directly with the requirements of the identified problem space..""")
        
    output_parser = JsonOutputParser(pydantic_object=OutputModel)
        
    prompt_template = PromptTemplate(
        template=PROMPT,
        input_variables=["description", "columns", "index"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.0)

    chain = prompt_template | llm | output_parser

    sample = json.loads(state["dataframe"].sample(5).to_json(orient="split"))

    response = chain.invoke({
        "description": state["description"],
        "columns": sample["columns"],
        "index": sample["index"]
    })

    return {"domain": response}


