# AI Vision

Streamlit and TensorFlow dog-versus-cat image classifier.

## Local setup

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep `dog_cat_model.keras` in the project root beside `app.py`.

## Streamlit Community Cloud

Select Python 3.11 in Advanced settings, set `app.py` as the main file,
and use `requirements.txt` for dependencies.
