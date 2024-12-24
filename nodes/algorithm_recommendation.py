from typing import Literal
from pydantic import BaseModel, Field
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from states.state import OverallState


def dataset_statistics(dataframe: pd.DataFrame, target_column: str):
    """
    Summarize the dataset
    Parameters
    ----------
    dataframe : pd.DataFrame
    The dataframe to summarize
    target_column : str
    The target column for the dataset
    Returns
    -------
    dict
    A dictionary containing a summary of the dataset
    """
    numerical_cols = dataframe.select_dtypes(include=["number"]).columns
    non_numerical_cols = dataframe.select_dtypes(exclude=["number"]).columns

    n_samples = dataframe.shape[0]
    n_features = dataframe.shape[1] - 1 
    missing_values = dataframe.isnull().sum().sum() > 0

    if target_column in numerical_cols:
        correlations = dataframe[numerical_cols].corr()[target_column]
        high_correlation = correlations.abs().max() > 0.5
    else:
        high_correlation = False

    if target_column in numerical_cols:
        target_distribution = ("Skewed" if dataframe[target_column].skew() > 0.5 else "Normal")
    else:
        target_distribution = "Categorical"

    high_dimensional = n_features > n_samples
    unique_counts = {col: dataframe[col].nunique() for col in non_numerical_cols if col != target_column}
    frequent_categories = {col: dataframe[col].value_counts().idxmax() for col in non_numerical_cols if col != target_column}

    dataset_summary = {
        "n_samples": n_samples,
        "n_features": n_features,
        "missing_values": missing_values,
        "feature_correlation": "High" if high_correlation else "Low",
        "target_distribution": target_distribution,
        "high_dimensional": high_dimensional,
        "non_numerical_features": {"count": len(non_numerical_cols),"unique_counts": unique_counts,"most_frequent_categories": frequent_categories},
        }

    return dataset_summary

REG_PROMPT = """
        I have a dataset with the following **Description** provided by the user:  `{description}` 
        The Domain of the dataset has been classified as `{domain}` and has the structure as follows: `{structure}`
        The dataset is summarized with the following **Statistics**: `{summary}`  

        ### **Task**  
        Based on the dataset **Description** and **Statistics**, 
        recommend appropriate regression algorithms that are suitable for the given dataset.
          Provide reasoning for each suggestion based on the dataset properties.  

        ### **Guidelines for Recommendations**  
        Consider the following factors when analyzing the dataset and suggesting regression models:  

        1. **Linear vs. Non-Linear Relationships**  
            - Suggest linear models (e.g., Linear Regression, Ridge, Lasso, ElasticNet) 
                if the relationship between features and the target is expected to be linear.  
            - Suggest non-linear models (e.g., RandomForestRegressor, SVR) if the relationship is expected to be complex or non-linear.  

        2. **Presence of Outliers**  
            - If the dataset has significant outliers, recommend robust regression models (e.g., Huber, RANSAC) 
                that can handle them effectively.  

        3. **Handling Missing Values**  
            - For datasets with missing values, consider preprocessing options (e.g., imputation) 
                or suggest models that can natively handle missing data if applicable.  

        4. **Skewed Target Distribution**  
            - If the target variable has a skewed distribution, suggest models that handle non-normal distributions 
                (e.g., QuantileRegressor) or recommend transformation techniques for the target variable (e.g., log transformation).  

        5. **Feature Correlation**  
            - For datasets where features have a high correlation with the target variable, 
                suggest regularization methods (e.g., Ridge, Lasso, ElasticNet) to improve model performance and generalization.  

        6. **High Dimensionality**
            - If the number of features is significantly greater than the number of samples, recommend models that perform well in high-dimensional spaces (e.g., Lasso, ElasticNet, or SGDRegressor).  
        
        ### **Output Format**:  
        Provide your response in the following JSON format:
        {format_instructions}"""

CLA_PROMPT = """
        I have a dataset with the following **Description** provided by the user: `{description}`  
        The Domain of the dataset has been classified as `{domain}` and has the structure as follows: `{structure}`
        The dataset is summarized with the following **Statistics**: `{summary}`  

        ### **Task**  
        Based on the dataset **Description** and **Statistics**, 
        recommend appropriate classification algorithms that are suitable for the given dataset.
        Make sure that those algorithm are available in the scikit-learn library.
        Provide reasoning for each suggestion based on the dataset properties.  

        ### **Guidelines for Recommendations**  
        Consider the following factors when analyzing the dataset and suggesting classification models:  

        1. **Linear vs. Non-Linear Decision Boundaries**  
            - Suggest linear models (e.g., Logistic Regression, LinearSVC) if the decision boundary between classes is expected to be linear.  
            - Suggest non-linear models (e.g., RandomForestClassifier, GradientBoostingClassifier, SVM with RBF kernel) if the decision boundary is complex or non-linear.  

        2. **Class Imbalance**  
            - If the dataset has imbalanced classes, recommend models that handle class imbalance (e.g., RandomForestClassifier, GradientBoostingClassifier)
              or suggest class weighting or oversampling techniques.  

        3. **Presence of Outliers**  
            - For datasets with significant outliers, suggest robust classification models (e.g., Robust Logistic Regression).  

        4. **Handling Missing Values**  
            - If the dataset contains missing values, suggest preprocessing techniques (e.g., imputation) 
            or models that natively handle missing data (e.g., XGBoost, CatBoost).  

        5. **Feature Dimensionality**  
            - For high-dimensional datasets, recommend models that perform well with many features (e.g., Logistic Regression with L1 penalty, LinearSVC).  
            - For datasets with more samples than features, suggest algorithms that leverage the abundance of data (e.g., RandomForestClassifier, Neural Networks).  

        6. **Scalability**  
            - For large datasets, suggest computationally efficient models (e.g., Naive Bayes, SGDClassifier, LinearSVC).  

        7. **Probabilistic Outputs**  
            - If the task requires class probabilities, recommend models that provide probabilistic outputs (e.g., Logistic Regression, Naive Bayes, GradientBoostingClassifier).  

        ### **Output Format**:  
        Provide your response in the following JSON format:
        {format_instructions}
        """

AD_PROMPT = """
        # Anomaly Detection Algorithm Recommendation Prompt  

        ### **Context**  
        I have a dataset with the following details:  
        - **Description**: `{description}`  
        - **Domain**: `{domain}`  
        - **Structure**: `{structure}`  
        - **Summary Statistics**: `{summary}`  

        ### **Task**  
        Based on the dataset's **Description** and **Statistics**, recommend appropriate **anomaly detection algorithms** that are suitable for the given dataset.  
        Make sure that these algorithms are available in the **scikit-learn library**.  
        Provide reasoning for each suggestion based on the dataset properties.  

        ### **Guidelines for Recommendations**  
        Consider the following factors when analyzing the dataset and suggesting classification models:  

        1. **Linear vs. Non-Linear Anomaly Separation**  
        - Recommend **linear models** (e.g., `EllipticEnvelope`) if the anomalies are separable from 
            the normal data using linear boundaries.  
        - Recommend **non-linear models** (e.g., `IsolationForest`, `OneClassSVM` with RBF kernel, `LocalOutlierFactor`) 
            for datasets with complex or non-linear separation of anomalies.  

        2. **Density-Based Anomaly Detection**  
        - Use models like `LocalOutlierFactor` if the anomalies are based on density differences relative to their neighbors.  

        3. **Presence of Outliers**  
        - Suggest robust models like `IsolationForest` or `RobustCovariance` (`EllipticEnvelope`) to handle datasets with significant outliers.  

        4. **Handling Missing Values**  
        - If missing values are present:  
            - Recommend preprocessing strategies like imputation.  
            - Suggest algorithms that can tolerate missing data after preprocessing.  

        5. **Feature Dimensionality**  
        - For **high-dimensional datasets**, recommend scalable models like `IsolationForest` or `OneClassSVM` with appropriate kernel adjustments.  
        - For datasets with **fewer features**, consider simpler models like `EllipticEnvelope` or statistical methods.  

        6. **Scalability**  
        - For large datasets, prioritize computationally efficient models like `IsolationForest` or approximate methods for anomaly detection.  

        7. **Model Flexibility**  
        - For datasets requiring adaptive anomaly detection, suggest models that can dynamically adjust to changing data distributions (e.g., `IsolationForest`).  

        ### **Output Format**:  
        Provide your response in the following JSON format:
        {format_instructions}
        """


llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.0)

def regression_algorithm_selection(state: OverallState):
    """
    This function is used to select the regression algorithms.
    It identifies the regression algorithms suitable for the dataset and provides a reason for the algorithm choice.
    """

    class OutputModel(BaseModel):
        """
        The output model for the regression algorithms.
        """
        algorithms: list = Field(
            ..., 
            description="List of algorithms that can be used for the given dataset."
        )
        cause: str = Field(
            ..., 
            description="Reasons behind the selection of the algorithms, based on the dataset properties. Keep it crisp and clear."
        )

    output_parser = JsonOutputParser(pydantic_object=OutputModel)

    prompt_template = PromptTemplate(
        template=REG_PROMPT,
        input_variables=["description", "summary", "domain", "task"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    chain = prompt_template | llm | output_parser

    summary = dataset_statistics(state["dataframe"], state["task"]["target"])

    response = chain.invoke({
        "description": state["description"],
        "summary": summary,
        "domain": state["domain"]["type"],
        "structure": state["structure"]["type"]
    })

    return {"methods": response}

def classification_algorithm_selection(state: OverallState):
    """
    This function is used to select the classification algorithms.
    It identifies the classification algorithms suitable for the dataset and provides a reason for the algorithm choice.
    """

    class OutputModel(BaseModel):
        """
        The output model for the classification algorithms.
        """
        algorithms: list = Field(
            ..., 
            description="List of algorithms suitable for the given dataset."
        )
        cause: str = Field(
            ..., 
            description="The reasoning for selecting the classification algorithms based on dataset properties."
        )

    output_parser = JsonOutputParser(pydantic_object=OutputModel)

    prompt_template = PromptTemplate(
        template=CLA_PROMPT,
        input_variables=["description", "summary", "domain", "task"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    chain = prompt_template | llm | output_parser

    summary = dataset_statistics(state["dataframe"], state["task"]["target"])

    response = chain.invoke({
        "description": state["description"], 
        "summary": summary,
        "domain": state["domain"]["type"],
        "structure": state["structure"]["type"]
    })

    return {"methods": response}

def anomaly_detection_algorithm_selection(state: OverallState):
    """
    This function is used to select the anomaly detection algorithms.
    It identifies the anomaly detection algorithms suitable for the dataset and provides a reason for the algorithm choice.
    """

    class OutputModel(BaseModel):
        """
        The output model for the anomaly detection algorithms.
        """
        algorithms: list = Field(
            ..., 
            description="List of algorithms suitable for the given dataset."
        )
        cause: str = Field(
            ..., 
            description="The reasoning for selecting the anomaly detection algorithms based on dataset properties."
        )

    output_parser = JsonOutputParser(pydantic_object=OutputModel)

    prompt_template = PromptTemplate(
        template=AD_PROMPT,
        input_variables=["description", "summary", "domain", "task"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    chain = prompt_template | llm | output_parser

    summary = dataset_statistics(state["dataframe"], state["task"]["target"])

    response = chain.invoke({
        "description": state["description"], 
        "summary": summary,
        "domain": state["domain"]["type"],
        "structure": state["structure"]["type"]
    })

    return {"methods": response}


def algorithm_selection_branch(state) -> Literal["Regression", "Classification", "Anomaly_Detection", END]:
    """
    This function is used to determine the task of the dataset.
    Based on the task type in the state, it returns the corresponding machine learning task category.
    """
    if state["task"]["type"] == "regression":
        return "Regression"
    
    if state["task"]["type"] == "classification":
        return "Classification"
    
    if state["task"]["type"] == "anomaly detection":
        return "Anomaly_Detection"
    
    return END


