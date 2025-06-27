Setup Instructions

    Clone the repository:

'''

git clone https://github.com/your-username/anpr-parking-system.git
cd anpr-parking-system/backend
'''bash

    Create and activate virtual environment:

'''

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
'''bash

    Install dependencies:

'''

pip install -r requirements.txt
'''bash


    Run the application:

'''

PYTHONPATH=. uvicorn main:app --reload --host 0.0.0.0 --port 8000
'''bash
