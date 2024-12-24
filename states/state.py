from typing import Annotated,Optional
from typing_extensions import TypedDict
from operator import add

class OverallState(TypedDict):
    """
    Overall state of Langgraph
    """
    dataframe : Optional[any]
    description : Optional[str]
    domain : Optional[dict]
    task : Optional[dict]
    methods : Optional[dict]
    structure : Optional[dict]
    statistics : Optional[any]
    columns : list[str]
    timestamp : Optional[str]
    configs : Annotated[list, add]
    pipeline : Optional[dict]

class ProcessingState(TypedDict):
    """
    Processing state of the data
    """
    statistics : Optional[any]
    domain: Optional[any]
    task: Optional[any]
    structure: dict
    column : str
    algorithms : list[str]
