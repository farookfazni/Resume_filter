# Resume Filter

## Project Overview

Resume Filter is a machine learning project that predicts the role of a candidate based on their resume. The model utilizes the Hugging Face pre-trained model `fazni/distilbert-base-uncased-career-path-prediction` and is trained on the `fazni/role-based-on-skills-2.0` dataset.

The model is deployed and running on the Hugging Face model hub in the `fazni/Resume-filter-plus-QA-documents` space. You can check out the live model and explore its predictions.

## Model Details

- **Pre-trained Model:** fazni/distilbert-base-uncased-career-path-prediction
- **Training Dataset:** fazni/role-based-on-skills-2.0
- **Live Model:** [Resume Filter on Hugging Face](https://huggingface.co/fazni/Resume-filter-plus-QA-documents)

## Q&A Documents

In addition to the role prediction model, the project also incorporates a set of Q&A documents using the OpenAI API. This feature enhances the understanding and interaction capabilities of the model, providing more context and information related to candidate resumes.

## Getting Started

To use the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/farookfazni/Resume_filter.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

   This will launch the application locally, and you can access it in your web browser.

## Usage

Visit the local URL provided by Streamlit (usually [http://localhost:8501](http://localhost:8501)) in your web browser. Upload a resume, and the model will predict the candidate's role.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository and submit a pull request.

## Issues and Bug Reports

If you encounter any issues or bugs, please open an issue on the [GitHub Issues](https://github.com/farookfazni/Resume_filter/issues) page.

## License

This project is licensed under the [MIT License](LICENSE).

Happy filtering!
