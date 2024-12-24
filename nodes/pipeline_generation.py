import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,\
                            IsolationForest, RandomForestRegressor,\
                            GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler,\
                                RobustScaler,OneHotEncoder, \
                                LabelEncoder,PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from states.state import OverallState


class FBFillTransformer(BaseEstimator, TransformerMixin):
    """
    Forward and backward fill transformer
    This transformer fills missing values in a dataset using forward and backward fill.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.fillna(method='ffill').fillna(method='bfill').values	


PROMPT = """
            ### **Task**
            From the list of function given below, select the function that can be used for the given query : {query}

            **Data processing functions**:
            - MeanImputer
            - MedianImputer
            - MostFrequentImputer
            - KNNImputer
            - LogisticRegression
            - RandomForestClassifier
            - GradientBoostingClassifier
            - LinearSVC
            - StandardScaler
            - MinMaxScaler
            - RobustScaler
            - OneHotEncoder
            - LabelEncoder
            - PowerTransformer
            - Box-CoxTransformer
            - QuantileTransformer
            - FBFillTransformer
            - No_Suitable_Function ( if the function is not available for the given query)

            **Regression functions**:
            - ElasticNet
            - Lasso
            - Ridge
            - LinearRegression
            - RandomForestRegressor
            - SVR

            **Classification functions**:
            - RandomForestClassifier
            - GradientBoostingClassifier
            - KNeighborsClassifier
            - LinearSVC
            - SVC
            - LogisticRegression
            - RidgeClassifier

            **Anomaly Detection functions**:
            - IsolationForest
            - OneClassSVM
            - LocalOutlierFactor
            - EllipticEnvelope

            return the function name. if none of the function is suitable for the given query
            {format_instructions}
            """

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.0)

def pipeline_generation(state: OverallState):
    """
    This function is used to map and reduce the data processing configuration.
    """
    class OutputModel(BaseModel):
        """ The output model for the data processing configuration."""
        function: str = Field(..., description=""" function that are avialable in the tools.""")

    parser = JsonOutputParser(pydantic_object=OutputModel)

    prompt_template = PromptTemplate(template=PROMPT,input_variables=["query"],partial_variables={"format_instructions": parser.get_format_instructions()})
    
    chain = prompt_template | llm | parser
    final_config = {}

    for i in state["configs"]:
        algorithm = i["algorithm"]
        response = chain.invoke({"query": algorithm})
        alg_func = response["function"]

        if alg_func not in final_config and alg_func != "No_Suitable_Function":
            final_config[alg_func] = {}
        elif alg_func in final_config and alg_func != "No_Suitable_Function":
            pass
        else:
            continue
        
        for method in i["methods"]:
            response = chain.invoke({"query": method})
            met_func = response["function"]
            if met_func not in final_config[alg_func] and met_func != "No_Suitable_Function":
                final_config[alg_func][met_func] = [i['column']]
            elif met_func in final_config[alg_func] and met_func != "No_Suitable_Function":
                final_config[alg_func][met_func].append(i['column'])
            else:
                pass
    
    pipelines = {}

    for model_name, preprocessing in final_config.items():
        data_preprocessing = []
        if "StandardScaler" in preprocessing:
            data_preprocessing.append(("StandardScaler", StandardScaler(), preprocessing["StandardScaler"]))

        if "MinMaxScaler" in preprocessing:
            data_preprocessing.append(("MinMaxScaler", MinMaxScaler(), preprocessing["MinMaxScaler"]))

        if "RobustScaler" in preprocessing:
            data_preprocessing.append(("RobustScaler", RobustScaler(), preprocessing["RobustScaler"]))
        
        if "OneHotEncoder" in preprocessing:
            data_preprocessing.append(("OneHotEncoder", OneHotEncoder(), preprocessing["OneHotEncoder"]))

        if "LabelEncoder" in preprocessing:
            data_preprocessing.append(("LabelEncoder", LabelEncoder(), preprocessing["LabelEncoder"]))

        if "PowerTransformer" in preprocessing:
            data_preprocessing.append(("PowerTransformer", PowerTransformer(), preprocessing["PowerTransformer"]))

        if "Box-CoxTransformer" in preprocessing:
            data_preprocessing.append(("Box-CoxTransformer", PowerTransformer(method="box-cox"), preprocessing["Box-CoxTransformer"]))

        if "QuantileTransformer" in preprocessing:
            data_preprocessing.append(("QuantileTransformer", PowerTransformer(method="quantile"), preprocessing["QuantileTransformer"]))

        if data_preprocessing:
            data_preprocessing = ColumnTransformer(transformers=data_preprocessing, remainder="passthrough")
        else:
            data_preprocessing = "passthrough"


        data_imputing = []
        if "MeanImputer" in preprocessing:
            data_imputing.append(("MeanImputer", SimpleImputer(strategy="mean"), preprocessing["MeanImputer"]))

        if "MedianImputer" in preprocessing:
            data_imputing.append(("MedianImputer", SimpleImputer(strategy="median"), preprocessing["MedianImputer"]))

        if "MostFrequentImputer" in preprocessing:
            data_imputing.append(("MostFrequentImputer", SimpleImputer(strategy="most_frequent"), preprocessing["MostFrequentImputer"]))
        
        if "KNNImputer" in preprocessing:
            data_imputing.append(("KNNImputer", KNNImputer(), preprocessing["KNNImputer"]))

        if "FBFillTransformer" in preprocessing:
            data_imputing.append(("FBFillTransformer", FBFillTransformer(), preprocessing["FBFillTransformer"]))

        if data_imputing:
            data_imputer = ColumnTransformer(transformers=data_imputing, remainder="passthrough")
        else:
            data_imputer = "passthrough"

        # Classification
        if model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif model_name == "LinearSVC":
            model = LinearSVC(C=1.0, max_iter=1000, random_state=42)

        # Regression
        elif model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "Lasso":
            model = Lasso(alpha=1.0)
        elif model_name == "ElasticNet":
            model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "GradientBoostingRegressor":
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif model_name == "SVR":
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

        # Anomaly Detection
        elif model_name == "OneClassSVM":
            model = OneClassSVM(kernel='rbf', gamma='scale')
        elif model_name == "IsolationForest":
            model = IsolationForest(random_state=42)
        elif model_name == "LocalOutlierFactor":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        elif model_name == "EllipticEnvelope":
            model = EllipticEnvelope(contamination=0.1, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        pipeline = Pipeline(steps=[
            ("data_preprocessing", data_preprocessing),
            ("Imputing", data_imputer),
            ("model", model)
        ])
        
        pipelines[model_name] = pipeline

    return {"pipeline": pipelines}