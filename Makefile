.PHONY: install run app audit clean
install:   ; pip install -r requirements.txt
run:       ; python -m src.pipeline
audit:     ; python scripts/signal_audit.py
app:       ; streamlit run app/streamlit_app.py
clean:     ; rm -rf artifacts/* outputs/* __pycache__ src/__pycache__
