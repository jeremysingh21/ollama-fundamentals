import os
from dotenv import load_dotenv
import pyodbc

# Load environment variables from .env file
load_dotenv()

try:
    conn = pyodbc.connect(
        f"DRIVER={os.getenv('DB_DRIVER')};SERVER={os.getenv('DB_SERVER')};DATABASE={os.getenv('DB_NAME')};UID={os.getenv('DB_USERNAME')};PWD={os.getenv('DB_PASSWORD')}"
    )
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {str(e)}")