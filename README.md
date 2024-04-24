# Text Embedding and Similarity Analysis Project with Transformers

This project contains a Python script that performs text embedding and similarity analysis operations using the Transformers library provided by Hugging Face. The project primarily converts texts into embedding vectors using the AutoTokenizer and AutoModelForCausalLM classes, and then measures the similarity between these vectors.

## Installation

To run the project, follow these steps:

1. Install the required libraries:
    ```
    pip install transformers scipy numpy torch pandas matplotlib scikit-learn
    ```

2. Start the project by running the `main.py` file:
    ```
    python main.py
    ```

## Files and Folders

- **main.py**: The main Python script that executes text embedding and similarity analysis operations.
- **app.log**: Log file containing log messages recorded during the process.
- **data/**: Folder containing sample data files.
- **embeddings/**: Folder where computed embedding vectors are stored.
- **plots/**: Folder where visualizations are saved.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
