import transformers

def load_models():
    # English to Tagalog
    EN_TL_MODEL = "Helsinki-NLP/opus-mt-en-tl"
    en_tl_tokenizer = transformers.AutoTokenizer.from_pretrained(EN_TL_MODEL)
    en_tl_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(EN_TL_MODEL)
    en_tl_translator = transformers.pipeline("text2text-generation", model=en_tl_model, tokenizer=en_tl_tokenizer, device=0)
    # Tagalog to English
    TL_EN_MODEL = "Helsinki-NLP/opus-mt-tl-en"
    tl_en_tokenizer = transformers.AutoTokenizer.from_pretrained(TL_EN_MODEL)
    tl_en_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(TL_EN_MODEL)
    tl_en_translator = transformers.pipeline("text2text-generation", model=tl_en_model, tokenizer=tl_en_tokenizer, device=0)
    return en_tl_translator, tl_en_translator
    
en_tl_translator, tl_en_translator = load_models()
