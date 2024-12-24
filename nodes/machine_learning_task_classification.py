from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from states.state import OverallState

PROMPT = """
        ### **Dataset Data Science Task Identification Prompt**  

        You are a skilled data analyst specializing in identifying problem statements from datasets.  
        Analyze the provided information to determine the type of problem (e.g., classification, regression, anomaly detection, forecasting).  

        ---

        ### **Inputs**:  
        1. **Human Expert Description**:  
        - "{description}"  
        - A summary from a **Domain** expert describing the dataset belonging to a "{domain}" (e.g., finance, healthcare, marketing).  

        2. **Dataset Metadata**:  
        - **Columns**: "{columns}" (features and structure of the dataset).  
        - **Index**: "{index}" (row indexing information).  
        - **Data Types**: "{dtypes}" (data types of the columns).  
        - **Columns Uniqueness**: "{columns_uniqueness}" (basic statistics of the dataset showing the percentage of unique values per column).  

        ---

        ### **Tasks**:  
        - Analyze the **Human Expert Description** to understand the dataset's purpose.  
        - If unclear or ambiguous, evaluate the **Dataset Metadata** (**Columns**, **Index**, or **Data Types**) for task identification.  
        - Use the following rules to classify the task:  

        1. **Classification or Regression (Supervised Learning)**:  
        - Determine whether the dataset includes a **target column** (e.g., "target," "label," "dependent," or "y").  
        - Use the **Data Types** and **Columns Uniqueness** of the target column to differentiate between classification and regression tasks:  
            - **Classification**: The target column has:  
            - A categorical or discrete data type (`object`, `category`, or `int64` with a limited number of unique values).  
            - A **Columns Uniqueness** of less than or equal to 10% (indicating discrete classes).  
            - **Regression**: The target column has:  
            - A continuous numerical data type (`float64` or `int64` with more than 10 unique values).  
            - A **Columns Uniqueness** greater than 10% (indicating continuous data).  

        2. **Forecasting**:  
        - The dataset contains **Columns** related to time (e.g., "date," "time," "timestamp") or terms like "trend," "seasonality," "cycle," "forecast," or "future."  

        3. **Anomaly Detection**:  
        - The dataset mentions "anomaly," "outlier," "deviation," "abnormal," or similar terms.  

        4. **Default**:  
        - If no clear determination can be made, default to **Anomaly Detection**.  

        ---

        ### **Output Format**: 
        {format_instructions}"""


def task_classification(state: OverallState):
    """
    This function is used to classify the machine learning task.
    It identifies the machine learning task and provides a reason for the task choice.
    Examples of tasks include classification, regression, anomaly detection, and forecasting.
    """

    class OutputModel(BaseModel):
        """
        The output model for the task classification.
        """
        type: Literal["classification", "regression", "anomaly detection", "forecasting"] = Field(
            ..., 
            description="The identified machine learning task, such as classification, regression, anomaly detection, or forecasting."
        )
        target: str = Field(
            ..., 
            description="For classification or regression tasks, provide the target column name. For other tasks, return an empty string."
        )
        cause: str = Field(
            ..., 
            description="Provide reasons for the task selection. Keep it concise and clear, based on dataset characteristics like keywords, unique identifiers, or domain-specific terminology."
        )

    output_parser = JsonOutputParser(pydantic_object=OutputModel)

    prompt_template = PromptTemplate(
        template=PROMPT,
        input_variables=["description", "columns", "index", "dtypes", "domain", "columns_uniqueness"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    chain = prompt_template | llm | output_parser

    sample = state["dataframe"].sample(5)

    response = chain.invoke({
        "description": state["description"],
        "columns": sample.columns.to_list(),
        "index": sample.index.to_list(),
        "dtypes": sample.dtypes.astype(str).to_dict(),
        "columns_uniqueness": state["statistics"].loc["Percentage of Unique Values"].to_dict(),
        "domain": state["domain"]["type"]
    })

    return {"task": response}



