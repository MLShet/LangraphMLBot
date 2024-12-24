import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END
from nodes.domain_identification import domain_identification
from nodes.dataset_type_detection import dataset_type_detection
from nodes.machine_learning_task_classification import task_classification
from nodes.data_preprocessing import data_preprocessing, fan_out_data_preprocessing
from nodes.pipeline_generation import pipeline_generation
from nodes.data_statistics_analysis import data_statistics_analysis
from nodes.algorithm_recommendation import regression_algorithm_selection, \
                                            classification_algorithm_selection, \
                                            anomaly_detection_algorithm_selection, \
                                            algorithm_selection_branch
from states.state import OverallState
import argparse
import pickle

def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def setup_graph():
    """
    Sets up the StateGraph and connects nodes with edges.
    """
    # Initialize the StateGraph with OverallState
    DataSet = StateGraph(OverallState)

    # Add nodes to the graph
    DataSet.add_node("Domain_Identification", domain_identification)
    DataSet.add_node("Dataset_Type_Detection", dataset_type_detection)
    DataSet.add_node("Machine_Learning_\nTask_Classification", task_classification)
    DataSet.add_node("Data_Statistics_Analysis", data_statistics_analysis)
    DataSet.add_node("Regression", regression_algorithm_selection)
    DataSet.add_node("Classification", classification_algorithm_selection)
    DataSet.add_node("Anomaly_Detection", anomaly_detection_algorithm_selection)
    DataSet.add_node("Data_Preprocessing", data_preprocessing)
    DataSet.add_node("Pipeline_Generation", pipeline_generation)

    # Add edges between nodes
    DataSet.add_edge(START, "Domain_Identification")
    DataSet.add_edge("Domain_Identification", "Dataset_Type_Detection")
    DataSet.add_edge("Dataset_Type_Detection", "Data_Statistics_Analysis")
    DataSet.add_edge("Data_Statistics_Analysis", "Machine_Learning_\nTask_Classification")

    # Conditional edges for machine learning tasks
    DataSet.add_conditional_edges("Machine_Learning_\nTask_Classification", algorithm_selection_branch, 
                                  ["Regression", "Classification", "Anomaly_Detection", END])
    DataSet.add_conditional_edges("Classification", fan_out_data_preprocessing, ["Data_Preprocessing"])
    DataSet.add_conditional_edges("Regression", fan_out_data_preprocessing, ["Data_Preprocessing"])
    DataSet.add_conditional_edges("Anomaly_Detection", fan_out_data_preprocessing, ["Data_Preprocessing"])
    DataSet.add_edge("Data_Preprocessing", "Pipeline_Generation")
    DataSet.add_edge("Pipeline_Generation", END)

    return DataSet


def main():
    """
    Main function to load data, set up the graph, and invoke the graph.
    """
    # Argument parser for input CSV file and description
    parser = argparse.ArgumentParser(description="Process a dataset and generate a machine learning pipeline.")
    parser.add_argument('csv_file', type=str, help='Path to the CSV dataset file')
    parser.add_argument('description', type=str, help='Short description of the dataset')

    args = parser.parse_args()

    # Load the dataset
    df = load_data(args.csv_file)

    # Set up the graph
    DataSet = setup_graph()

    # Compile the graph and execute
    app = DataSet.compile()
    print("Graph compiled successfully.")
    # Invoke the graph with the provided description and dataframe
    final_state = app.invoke({
        "description": args.description,
        "dataframe": df
    })
    print("Pipeline generation completed successfully.")
    print("Final pipeline state:")
    print(final_state)

    with open("pipelines.pkl", 'wb') as f:
        pickle.dump(final_state["pipeline"], f)

    print("Graph state saved to pipelines.pkl")

if __name__ == "__main__":
    main()



