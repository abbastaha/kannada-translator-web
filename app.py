import os
from flask import Flask, render_template, request
import torch
#twransformer from facebook NLLB
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import single_detection
# Import the transliteration library
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

app = Flask(__name__)

#this code will be enhanced to add multilang support, since my machine does not have aGPU,
#  the model is pretty hevy to load , you can change the model to another more heavier one for better translation

# --- Model Loading (NLLB) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "facebook/nllb-200-distilled-600M"
print(f"Loading tokenizer: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Loading model: {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
print("NLLB model loaded.")
NLLB_LANG_KANNADA = "kan_Latn"
NLLB_LANG_ENGLISH = "eng_Latn"
# --- End Model Loading ---

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text_final = None # Renamed for clarity
    input_text = None
    detected_language_display = None

    if request.method == 'POST':
        input_text = request.form.get('text', '').strip()
        if input_text:
            try:
                # 1. Detect Language
                print("DEBUG: Detecting language...")
                API_KEY = os.getenv('API_KEY')
                #API KEY is set as environment variable, take API key from hugging face api website
                detected_lang_simple = single_detection(input_text, api_key=API_KEY)
                detected_lang_display_code = detected_lang_simple or 'Unknown'
                detected_language_display = f"Detected: {detected_lang_display_code.upper()}"
                print(f"DEBUG: Detected language code (simple): {detected_lang_simple}")

                target_lang_is_kannada = False # Flag to know if we need transliteration

                # 2. Determine Source and Target NLLB Codes
                if detected_lang_simple == 'kn':
                    print("DEBUG: Path for Kannada detected")
                    source_lang_code = NLLB_LANG_KANNADA
                    target_lang_code = NLLB_LANG_ENGLISH
                elif detected_lang_simple == 'en':
                    print("DEBUG: Path for English detected")
                    source_lang_code = NLLB_LANG_ENGLISH
                    target_lang_code = NLLB_LANG_KANNADA
                    target_lang_is_kannada = True # Set the flag
                else:
                    print(f"DEBUG: Path for unsupported language: {detected_lang_simple}")
                    translated_text_final = "Language not supported (currently only Kannada and English)"
                    return render_template('index.html',
                                           translated_text=translated_text_final, # Use final variable
                                           input_text=input_text,
                                           detected_language=detected_language_display)

                # 3. Tokenize
                print(f"DEBUG: Tokenizing for source: {source_lang_code}...")
                print('Inoput text is , ', input_text)
                tokenizer.src_lang = source_lang_code
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

                # 4. Generate Translation (Output is based on target_lang_code)
                print(f"DEBUG: Generating translation for target: {target_lang_code}...")
                target_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)
                print(f"DEBUG: Target Language ID: {target_lang_id}")

                # --- Verification (Optional but recommended to keep for now) ---
                kannada_id_check = tokenizer.convert_tokens_to_ids(NLLB_LANG_KANNADA)
                english_id_check = tokenizer.convert_tokens_to_ids(NLLB_LANG_ENGLISH)
                print(f"DEBUG: Verification - ID for '{NLLB_LANG_KANNADA}': {kannada_id_check}")
                print(f"DEBUG: Verification - ID for '{NLLB_LANG_ENGLISH}': {english_id_check}")
                # --------------------------------------------------------------

                with torch.no_grad():
                    generated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=target_lang_id,
                        max_length=512
                    )

                # 5. Decode NLLB Output (This will be in Latin script if target was kan_Latn)
                print("DEBUG: Decoding generated tokens...")
                decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                nllb_output_text = decoded_outputs[0] if decoded_outputs else "" # Get the raw NLLB output
                print(f"DEBUG: NLLB Raw Decoded Output: {nllb_output_text}")


                # 6. Transliterate if Target was Kannada
                if target_lang_is_kannada and nllb_output_text:
                    print(f"DEBUG: Transliterating '{nllb_output_text}' from ITRANS to Kannada script...")
                    # Using ITRANS as the intermediate Roman scheme is common for this library
                    # It might require some experimentation to find the best source scheme (HK, ITRANS, etc.)
                    # that matches NLLB's Latin output style. ITRANS is a decent starting point.
                    try:
                        translated_text_final = transliterate(nllb_output_text, sanscript.ITRANS, sanscript.KANNADA)
                        print(f"DEBUG: Transliterated Output: {translated_text_final}")
                    except Exception as trans_err:
                        print(f"ERROR: Transliteration failed: {trans_err}")
                        # Fallback to showing the Latin script output if transliteration fails
                        translated_text_final = nllb_output_text + " (Transliteration failed)"
                else:
                    # If target was English or NLLB output was empty, use it directly
                    translated_text_final = nllb_output_text

                print("Translation and processing complete.")

            except Exception as e:
                print(f"Error during processing: {e}")
                import traceback
                traceback.print_exc()
                translated_text_final = f"An error occurred during translation. Please check server logs."

        else:
            translated_text_final = "Please enter text to translate."
            detected_language_display = None

        # Render template after POST processing
        return render_template('index.html',
                               translated_text=translated_text_final, # Pass final result
                               input_text=input_text,
                               detected_language=detected_language_display)

    # Render template for GET request
    return render_template('index.html',
                           translated_text=translated_text_final, # Pass final result
                           input_text=input_text,
                           detected_language=detected_language_display)


if __name__ == '__main__':
    print("Starting Flask app with NLLB model and Transliteration...")
    app.run(debug=True, host='127.0.0.1', port=5000)