setup:
\tpip install -r requirements.txt || true

backtest:
\tpython -m src.evaluation.backtest

serve:
\tuvicorn api.main:app --reload
