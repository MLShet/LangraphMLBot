# Automated ML Pipeline Builder

An AI-driven agent built with **LangChain** and **LangGraph**, powered by **ChatGPT**, to automate the creation of **Scikit-learn (Sklearn)** machine learning pipelines. The system processes CSV files and generates end-to-end data processing and model training pipelines that are ready for deployment.

---

## Features

- **Domain Identification**: Determines the domain of the dataset from metadata and user descriptions.
- **Dataset Type Detection**: Automatically identifies if the dataset is tabular or time series.
- **Task Classification**: Identifies the task type (classification, regression, or anomaly detection) based on the data and user-provided input.
- **Data Statistics Analysis**: Computes key dataset statistics to guide further processing and model selection.
- **Algorithm Recommendation**: Selects the most suitable algorithm from the Scikit-learn library based on the dataset's characteristics.
- **Data Preprocessing**: Automatically applies column-specific transformations, including data cleaning, imputation, scaling, and more.
- **Pipeline Generation**: Combines preprocessing, algorithm selection, and task-specific configurations into a complete Sklearn pipeline for training and deployment.

---

## System Workflow

1. **Input**:
   - Accepts a CSV file as input.
   - Optionally includes a description of the dataset for better task identification.

2. **Multi-Node Processing**:
   - **Domain Identification**: Identifies the problem domain.
   - **Dataset Type Detection**: Detects if the dataset is time series or tabular.
   - **Task Classification**: Classifies the task type (classification, regression, or anomaly detection).
   - **Data Statistics Analysis**: Computes key dataset statistics (e.g., missing values, distributions).
   - **Algorithm Recommendation**: Selects the best algorithm from Sklearn based on the dataset.
   - **Data Preprocessing**: Determines column-specific preprocessing steps.
   - **Pipeline Generation**: sklearn pipeline for the task with data processing units

3. **Output**:
   - Generates a complete Sklearn pipeline.
   - The pipeline includes preprocessing, algorithm application, and task-specific configurations.
   - Ready-to-train and deploy model.

---

## Installation

### Prerequisites
- Python 3.8+
- Pip or another package manager
- Basic understanding of machine learning concepts

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/MLShet/LangraphMLBot.git
   cd ai-ml-pipeline-builder
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install LangChain and LangGraph (if not included in requirements):
   ```bash
   pip install langchain langgraph
   ```

---

## Usage

1. Prepare your dataset as a CSV file.
2. Run the AI agent:
   ```bash
   python main.py --input <path_to_csv> --description "<optional_description>"
   ```
3. View the generated pipeline:
   - The output will include a serialized Sklearn pipeline that you can use directly for training or deployment.

---

## Example

Input CSV file:
```
feature1, feature2, target
1.2, 3.4, 1
2.3, 4.5, 0
...
```

Run the agent:
```bash
python main.py --input data.csv --description "Binary classification task"
```

Output (Example):
```python
Pipeline(steps=[
    ('preprocessing', ColumnTransformer(...)),
    ('model', LogisticRegression())
])
```

---

## Project Structure
```
.
├── README.md          # Project documentation
├── main.py            # Entry point for the AI agent
├── requirements.txt   # Required Python packages
├── nodes/             # Multi-node logic for pipeline generation
├── states/             # Multi-node logic for pipeline generation
├── examples/          # Example datasets and usage
└── tests/             # Unit tests for each node
```

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

### Steps to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork:
   ```bash
   git commit -m "Add feature-name"
   git push origin feature-name
   ```
4. Submit a pull request.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [LangChain](https://github.com/hwchase17/langchain) for workflow orchestration.
- [LangGraph](https://github.com/langgraph/langgraph) for advanced graph-based task flows.
- [Scikit-learn](https://scikit-learn.org) for machine learning capabilities.
- OpenAI's ChatGPT for natural language processing capabilities.

