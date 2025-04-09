# Kannada-English Translator (Opus Models)

A web-based MVP for Kannada <=> English translation using Facebook NLLB models via the Transformers library and Flask.

## Setup

1.  Clone the repository (or create the files as described).
2.  Navigate to the project directory: `cd kannada_translator_opus`
3.  Create and activate a virtual environment:
    `python3 -m venv venv`
    `source venv/bin/activate`
4.  Install dependencies:
    `pip install -r requirements.txt`

## Running the Application

1.  Ensure the virtual environment is activated.
2.  Run the Flask app:
    `python app.py`
3.  Open your web browser to `http://127.0.0.1:5000/`

## Notes

*   The first time you run the app, the `transformers` library will download the translation models (Helsinki-NLP/opus-mt-kn-en and Helsinki-NLP/opus-mt-en-kn), which may take a few moments.
*   Language detection uses the `deep_translator` library.