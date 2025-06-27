Setup Instructions

Clone the repository:

    git clone https://github.com/your-username/anpr-parking-system.git](https://github.com/chittinarora/IntelliVision/tree/master/ANPR)
    cd anpr-parking-system/backend

Create and activate virtual environment:

    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate    # Windows

Install dependencies:

    pip install -r requirements.txt


Run the application:

    PYTHONPATH=. uvicorn main:app --reload --host 0.0.0.0 --port 8000
