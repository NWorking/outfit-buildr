@echo off
rem Create a virtual environment and install dependencies
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo Setup complete. Activate the environment with "call venv\Scripts\activate" and run "python app.py".
