# 
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from states.state import OverallState


PROMPT = """
        ### **Task**  
        Classify a DataFrame as either a **Time Series** or a **Tabular Dataset** using its metadata.  

        ---

        ### **Inputs**  
        1. **Column Names**:  
        A list of column names in the DataFrame, provided as `{columns}`.  
        2. **Column Data Types**:  
        Data types of the columns in the DataFrame, represented as `{dtype}`.  
        3. **Index Values**:  
        Sample entries of the DataFrame's index, represented as `{index}`.  
        4. **Index Data Type**:  
        The data type of the index (`index.dtype`), represented as `{idtype}`.  

        ---

        **Classification Criteria**  

        **1. Time Series**  
        Classify the DataFrame as a time series if **any** of the following conditions are met:  
        - A **column name** contains time-related terms such as `"timestamp"`, `"datetime"`, `"date"`, or `"time"`.  
        - A **column data type** is time-related (e.g., `datetime64`, `timedelta64`).  
        - The **index** contains datetime-like values, or its data type (`idtype`) is time-related (e.g., `DatetimeIndex`, `datetime64[ns]`).  

        **2. Tabular**  
        If none of the above conditions for time series classification are satisfied, classify the DataFrame as **tabular**.  

        ### **Output Format**:  
        Provide your response in the following JSON format:
        {format_instructions}"""



def dataset_type_detection(state: OverallState):
    """
    This function is used to classify the structure of the dataset.
    It identifies the structure of the dataset and provides a reason for the structure choice.
    Examples of dataset structures include 'Time series' and 'Tabular'.
    """
    class OutputModel(BaseModel):
        """
        The output model for the structure classification.
        """
        type: Literal['Time series', 'Tabular'] = Field(
            ..., 
            description="The dataset structure classified as 'Time series' or 'Tabular'."
        )
        column: str = Field(
            ..., 
            description="The name of the column belonging to the time series structure. "
                        "If none, return an empty string."
        )
        cause: str = Field(
            ..., 
            description="A concise explanation for the structure classification."
        )

    output_parser = JsonOutputParser(pydantic_object=OutputModel)

    prompt_template = PromptTemplate(
        template=PROMPT,
        input_variables=["columns", "dtype", "index", "idtype"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    sample = state["dataframe"].sample(5)

    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.0)
    
    chain = prompt_template | llm | output_parser

    response = chain.invoke({
        "columns": sample.columns.to_list(),
        "dtype": sample.dtypes.to_list(),
        "index": sample.index.to_list(),
        "idtype": sample.index.dtype.name
    })

    return {"structure": response}


