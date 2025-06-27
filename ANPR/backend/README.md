Setup Instructions

    Clone the repository:

'''bash

git clone https://github.com/your-username/anpr-parking-system.git
cd anpr-parking-system/backend
'''

    Create and activate virtual environment:

'''bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
'''

    Install dependencies:

'''bash

pip install -r requirements.txt
'''


    Run the application:

'''bash

PYTHONPATH=. uvicorn main:app --reload --host 0.0.0.0 --port 8000
'''

