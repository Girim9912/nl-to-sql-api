from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os
import sqlite3
import shutil
import json
import uuid
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI



class QueryRequest(BaseModel):
    query: str


# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("DEEPINFRA_API_KEY")

if not api_key:
    raise RuntimeError("🔐 API key not found! Please check your .env file.")

# ✅ Connect to DeepInfra-compatible endpoint using OpenAI wrapper
client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=api_key,
)

# 🔧 Create FastAPI app
app = FastAPI()

# 🌐 Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)
# Allowed file types and size limit
ALLOWED_EXTENSIONS = {".csv", ".db", ".txt", ".xlsx", ".xls", ".sqlite", ".sqlite3"}
MAX_FILE_SIZE_MB = 10

def validate_upload(file: UploadFile):
    """Validate uploaded file's extension and size."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    contents = file.file.read()
    file.file.seek(0)
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail="File exceeds size limit of 10MB")


# Configuration
DEFAULT_DB_PATH = "chinook.db"  # Your default database
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/inference/mistralai/Mistral-7B-Instruct-v0.2"

# Helper functions for file processing
def process_csv(file_path: str, db_path: str) -> None:
    """Convert a CSV file to SQLite database."""
    try:
        df = pd.read_csv(file_path)
        conn = sqlite3.connect(db_path)
        # Use the filename (without extension) as the table name
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        # Clean up table name (remove invalid chars)
        table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

def process_excel(file_path: str, db_path: str) -> None:
    """Convert an Excel file to SQLite database."""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        conn = sqlite3.connect(db_path)
        
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            # Clean up table name (remove invalid chars)
            table_name = ''.join(c if c.isalnum() else '_' for c in sheet_name)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing Excel: {str(e)}")

def process_text(file_path: str, db_path: str) -> None:
    """Convert a text file to SQLite database."""
    try:
        # Basic processing - assume tab or comma separated
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to detect delimiter
        delimiters = [',', '\t', '|', ';']
        lines = content.strip().split('\n')
        
        if not lines:
            raise ValueError("Empty file")
        
        # Count occurrences of each delimiter in first line
        counts = {d: lines[0].count(d) for d in delimiters}
        delimiter = max(counts, key=counts.get) if any(counts.values()) else ','
        
        # Parse as CSV with the detected delimiter
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        conn = sqlite3.connect(db_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing text file: {str(e)}")

def get_table_info(db_path: str) -> List[dict]:
    """Get information about tables in the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_info = []
        
        for table in tables:
            table_name = table[0]
            if table_name.startswith('sqlite_'):  # Skip SQLite internal tables
                continue
                
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            
            # Sample data (first 5 rows)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            sample_data = cursor.fetchall()
            
            col_names = [col[1] for col in columns]
            sample_rows = []
            
            for row in sample_data:
                row_dict = {}
                for i, val in enumerate(row):
                    row_dict[col_names[i]] = val
                sample_rows.append(row_dict)
            
            table_info.append({
                "name": table_name,
                "columns": [{"name": col[1], "type": col[2]} for col in columns],
                "row_count": row_count,
                "sample_data": sample_rows
            })
        
        conn.close()
        return table_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading database: {str(e)}")

def get_database_schema(db_path: str) -> str:
    """Extract the schema from a SQLite database to provide to the LLM."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = []
        
        for table in tables:
            table_name = table[0]
            if table_name.startswith('sqlite_'):
                continue  # Skip SQLite internal tables
                
            # Get table structure
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            column_info = []
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                column_info.append(f"{col_name} ({col_type})")
            
            # Get sample data count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            
            schema_info.append(f"Table: {table_name} ({count} records)\nColumns: {', '.join(column_info)}")
        
        conn.close()
        return "\n\n".join(schema_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting schema: {str(e)}")

def nl_to_sql(query: str, db_schema: str) -> str:
    """Convert natural language to SQL using DeepInfra API."""
    try:
        prompt = f"""<s>[INST] You are an expert SQL query builder. 
        Convert the following natural language question into a valid SQL query.
        
        Here's the database schema information:
        {db_schema}
        
        Question: {query}
        
        Please provide only the SQL query without any explanations or markdown formatting. [/INST]"""
        
        # Call DeepInfra API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}"
        }
        
        payload = {
            "input": prompt,
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        response = requests.post(
            DEEPINFRA_API_URL,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
            
        result = response.json()
        sql_query = result.get('generated_text', '').strip()
        
        # Remove model's instruction formatting if present
        start_marker = "</s><s>[INST]"
        if start_marker in sql_query:
            sql_query = sql_query.split(start_marker)[0].strip()
            
        # Remove markdown code blocks if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
            
        return sql_query.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

@app.post("/upload-db")
async def upload_database(file: UploadFile = File(...)):
    """Handle file uploads and convert to SQLite if needed."""
    # Generate a unique filename
    validate_upload(file)

    file_uuid = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    
    # Save the uploaded file
    temp_file_path = f"uploads/{file_uuid}_temp{file_extension}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Path for the SQLite database
    db_path = f"uploads/{file_uuid}.db"
    
    try:
        # Process based on file type
        if file_extension in ['.csv']:
            process_csv(temp_file_path, db_path)
        elif file_extension in ['.xlsx', '.xls']:
            process_excel(temp_file_path, db_path)
        elif file_extension in ['.txt']:
            process_text(temp_file_path, db_path)
        elif file_extension in ['.db', '.sqlite', '.sqlite3']:
            # Just copy the file if it's already a SQLite database
            shutil.copy(temp_file_path, db_path)
        else:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Get database info
        table_info = get_table_info(db_path)
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return JSONResponse({
            "status": "success",
            "filename": original_filename,
            "database_path": db_path,
            "table_info": table_info
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(db_path):
            os.remove(db_path)
        raise HTTPException(status_code=500, detail=str(e))
def extract_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema_lines = []

    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        schema_lines.append(f"Table {table_name}: {', '.join(col_names)}")

    conn.close()
    return "Schema:\n" + "\n".join(schema_lines)

@app.post("/generate-sql")
async def generate_sql(req: QueryRequest):
    try:
        # 🔍 Locate the latest uploaded DB (you can improve this with session/user logic later)
        db_files = sorted(Path("uploads").glob("*.db"), key=os.path.getmtime, reverse=True)
        if not db_files:
            raise HTTPException(status_code=400, detail="No database uploaded yet.")
        
        db_path = str(db_files[0])

        # 🧠 Extract schema from SQLite database
        schema = extract_db_schema(db_path)

        # 🔥 Craft the prompt
        prompt = (
            f"You are a helpful assistant that generates SQL for SQLite.\n"
            f"{schema}\n"
            f"Translate the following request into SQL:\n{req.query}"
        )

        # 🧠 Send to DeepInfra
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are an expert SQL assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150,
        )

        sql = response.choices[0].message.content.strip()

        # 🔄 Clean response if wrapped in markdown
        if "```" in sql:
            sql = sql.split("```")[1].replace("sql", "").strip()

        # 🧪 Try executing
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        return {"sql": sql, "results": rows}

    except Exception as e:
        print("🔥 Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

    """Generate SQL from natural language query and execute it."""
    try:
        query = request.get("query")
        db_path = request.get("database_path", DEFAULT_DB_PATH)
        
        if not query:
            return JSONResponse({
                "error": "No query provided"
            })
            
        if not os.path.exists(db_path):
            return JSONResponse({
                "error": f"Database not found: {db_path}"
            })
        
        # Get database schema for the selected database
        db_schema = get_database_schema(db_path)
        
        # Generate SQL from natural language
        sql_query = nl_to_sql(query, db_schema)
        
        # Execute the query
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get column names
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch results
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            formatted_results = []
            for row in results:
                formatted_results.append(dict(zip(column_names, row)))
            
            conn.close()
            
            return {
                "sql": sql_query,
                "results": formatted_results
            }
        except sqlite3.Error as e:
            return {
                "sql": sql_query,
                "error": f"SQL execution error: {str(e)}"
            }
            
    except Exception as e:
        return {
            "error": f"Server error: {str(e)}"
        }
@app.get("/")
def root():
    return {"message": "✅ NLtoSQL backend is running"}

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)