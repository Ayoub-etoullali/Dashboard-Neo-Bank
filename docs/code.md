# Code

source .venv/bin/activate

## Create Explainer
python code/create_explainer.py

## API
cd code
uvicorn api:app --host 0.0.0.0 --port 8649

## Streamlit
streamlit run streamlit_app.py --server.port 4571