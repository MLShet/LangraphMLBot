from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.constants import Send
from langchain_openai import ChatOpenAI

from states.state import OverallState, ProcessingState

PROMPT ="""
    ### **Task**  
    Determine the most suitable **Data Processing Methods** for a given column based on the provided **Description**, **Statistics**, **Domain**, **Structure**, **Task**, and **Algorithm**.  
    ---
    ### **Inputs**:  
    1. **Description**: `{description}`  
    - A brief explanation of the dataset that may or may not contain column-level information.  

    2. **Domain**: `{domain}`  
    - The dataset's domain (e.g., finance, healthcare, retail).  

    3. **Statistics**: `{statistics}`  
    - A JSON string of dataset statistics that can be loaded into a pandas DataFrame. This includes metrics such as missing values, unique values, variance, mean, median, standard deviation, skewness, and IQR.  

    4. **Structure**: `{structure}`  
    - The type of dataset (e.g., "Time Series" or "Tabular").  

    5. **Column**: `{column}`  
    - The column name for which the processing methods must be evaluated.  

    6. **Task**: `{task}`  
    - The dataset's task (e.g., "classification," "regression," "anomaly detection," or "forecasting").  

    7. **Algorithm**: `{algorithm}`  
    - The algorithm recommended for the task, which may influence the choice of preprocessing methods (e.g., tree-based models, linear regression, SVM).  

    ---

    ### **Data Processing Units**:  

    1.  - **Forward/Backward Fill**: Use for sequential/time-ordered data with small gaps in forecasting or regression tasks.  
        - **Median Imputation**: Use for numerical columns based on the skewness or distribution of the column.  
        - **KNN Imputation Classification**: Use for categorical columns or numerical columns with strong modal behavior for classifucation tasks.
        - **KNN Imputation Regression**: Use for numerical columns with strong linear relationships for regression tasks. 
    - **Avoid**: If the column has a very high percentage of missing values and no meaningful imputation strategy exists.  

    2. - **Standardization**: Use for numerical columns in models sensitive to scale, like linear regression or SVM.  
        - **Min Max Scaling**: Use for numerical features in neural networks or distance-based models (e.g., KNN).  
        - **Robust Scaling**: Use for numerical features with outliers.  
    - **Avoid**: For scale-invariant models such as tree-based algorithms.  

    3.- **One Hot Encoding**: Use for nominal categorical variables with manageable unique values.  
        - **Label Encoding**: Use for ordinal categorical variables or high-cardinality categories in tree-based models. 

    4. **Feature Dropping** 
        Criteria:  
        - Extremely high missing values (e.g., >80%) and no suitable imputation method.  
        - Very low variance or redundancy with other columns (e.g., highly correlated with another feature).  
        - Irrelevance to the problem statement or the target variable.  

    5.- **Log Transformation**: Use for numerical features with skewed distributions.  
        - **Box Cox**: Use for non-normal distributions.  

    6.- **Skip Processing**  : If the data is already clean or transformations are unnecessary for the task or algorithm.  

    ---

    ### **Evaluation Criteria**:  

    1. **Multiple Steps**:  
    - Processing may involve multiple steps for a column, such as imputation followed by scaling. Ensure the order of operations is logical and beneficial for the task.  

    2. **Problem Statement Alignment**:  
    - Assess the column's relevance to the task.  

    3. **Domain Knowledge Integration**:  
    - Leverage domain insights to justify the choice of processing methods.  

    4. **Algorithm Compatibility**:  
    - Adapt preprocessing methods to the strengths and weaknesses of the recommended algorithm.  
    ---
    ### **Output Format**  
    Provide results in JSON format:
    Return your choice of methods and the justification in the following format:{format_instructions}"""

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.0)

def data_preprocessing(state: ProcessingState):
    """
    This function generates the data processing configuration based on the provided state.
    """
    if not all(key in state for key in ["statistics", "column", "domain", "task", "structure", "algorithms", "description"]):
        raise ValueError("Missing required keys in the state.")

    class OutputModel(BaseModel):
        """ The output model for the data processing configuration."""
        column: str = Field(..., description="Name of the column being processed.")
        methods: list[str] = Field(..., description="List of methods applied to the column.")
        algorithm: str = Field(..., description="The algorithm used for the method.")
        reason: str = Field(..., description="Short explanation of the reason for the choice of method for the column.")

    prompt = PromptTemplate(
        template=PROMPT,
        input_variables=["statistics", "domain", "structure", "column", "task"],
        partial_variables={"format_instructions": JsonOutputParser(pydantic_object=OutputModel).get_format_instructions()}
    )

    parser = JsonOutputParser(pydantic_object=OutputModel)

    chain = prompt | llm | parser

    response = chain.invoke({
        "statistics": state["statistics"],
        "column": state["column"],
        "domain": state["domain"],
        "task": state["task"],
        "structure": state["structure"],
        "algorithm": state["algorithms"],
        "description": state["description"]
    })

    return {"configs": [response]}


def fan_out_data_preprocessing(state: OverallState):
    """ This function is used to fan out the data processing configuration."""
    combi = [(col, algo) for col in state["statistics"].columns for algo in state["methods"]["algorithms"]]
    return [Send("Data_Preprocessing", {"statistics" : state["statistics"][col],
                                            "domain": state["domain"]["type"],
                                            "task": state["task"]["type"],
                                            "column": col,
                                            "structure" :state["structure"]["type"],
                                            "description": state["description"],
                                            "algorithms" : algo}) for col,algo in combi]