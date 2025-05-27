@echo off
echo 🚀 Starting AI Forex Signal Generator...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo 📋 Checking requirements...
pip install -r requirements.txt

REM Check if setup has been run
if not exist ".setup_complete" (
    echo ⚙️ Running initial setup...
    python setup.py
    echo. > .setup_complete
)

REM Start the application
echo 🌐 Starting Streamlit application...
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo ✅ Application started successfully!
echo 🌐 Open your browser and go to: http://localhost:8501

pause
