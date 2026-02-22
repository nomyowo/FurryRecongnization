import os
import shutil
import sqlite3
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, status, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename
import db

app = FastAPI()

# Configuration
UPLOAD_FOLDER_LIB = 'lib'
UPLOAD_FOLDER_QUERY = 'query'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER_LIB, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_QUERY, exist_ok=True)

# Mount static files to serve images directly
app.mount("/files/lib", StaticFiles(directory=UPLOAD_FOLDER_LIB), name="lib_files")
app.mount("/files/query", StaticFiles(directory=UPLOAD_FOLDER_QUERY), name="query_files")

templates = Jinja2Templates(directory="templates")

def get_db():
    conn = db.get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.on_event("startup")
def startup_event():
    conn = db.get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            type TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/lib")
def view_lib(request: Request, label: Optional[str] = Query(None), conn: sqlite3.Connection = Depends(get_db)):
    if label == "":
        label = None
    print(f"DEBUG: label='{label}', type={type(label)}")
    labels = conn.execute('SELECT label, COUNT(*) as count FROM images WHERE type = ? GROUP BY label ORDER BY label', ('lib',)).fetchall()
    images = []
    if label is not None:
        images = conn.execute('SELECT * FROM images WHERE type = ? AND label = ? ORDER BY id DESC', ('lib', label)).fetchall()
    print(f"DEBUG: images count={len(images)}")

    return templates.TemplateResponse("lib.html", {"request": request, "images": images, "labels": labels, "current_label": label})


@app.post("/lib/upload")
def upload_lib(
    request: Request,
    files: List[UploadFile] = File(...),
    label: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db)
):
    for file in files:
        if file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_name = f"{timestamp}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER_LIB, save_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            conn.execute('INSERT INTO images (filename, label, type, timestamp) VALUES (?, ?, ?, ?)',
                         (save_name, label, 'lib', str(datetime.now())))

    conn.commit()
    return RedirectResponse(url="/lib", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/query")
def view_query(request: Request, label: Optional[str] = Query(None), conn: sqlite3.Connection = Depends(get_db)):
    # Fetch distinct labels and counts
    # We treat empty string or NULL as "unlabeled"
    db_labels = conn.execute('SELECT label, COUNT(*) as count FROM images WHERE type = ? GROUP BY label ORDER BY label', ('query',)).fetchall()

    labels = []
    for row in db_labels:
        lbl = row['label']
        if not lbl: # Handle empty string or None
            labels.append({'label': '未检测到', 'original_label': '', 'count': row['count']})
        else:
            labels.append({'label': lbl, 'original_label': lbl, 'count': row['count']})

    images = []

    # Check if we are filtering by a specific label
    if label is not None:
        if label == "未检测到":
             # Special case: fetch images with empty label
             images = conn.execute("SELECT * FROM images WHERE type = ? AND (label = '' OR label IS NULL) ORDER BY id DESC", ('query',)).fetchall()
        else:
             images = conn.execute('SELECT * FROM images WHERE type = ? AND label = ? ORDER BY id DESC', ('query', label)).fetchall()

    return templates.TemplateResponse("query.html", {"request": request, "images": images, "labels": labels, "current_label": label})


@app.post("/query/upload")
def upload_query(
    request: Request,
    files: List[UploadFile] = File(...),
    label: Optional[str] = Form(""),
    conn: sqlite3.Connection = Depends(get_db)
):
    for file in files:
        if file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_name = f"{timestamp}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER_QUERY, save_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            conn.execute('INSERT INTO images (filename, label, type, timestamp) VALUES (?, ?, ?, ?)',
                         (save_name, label, 'query', str(datetime.now())))

    conn.commit()
    return RedirectResponse(url="/query", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/update_label/{id}")
def update_label(
    id: int,
    label: str = Form(...),
    redirect_to: str = Form("/"),
    conn: sqlite3.Connection = Depends(get_db)
):
    conn.execute('UPDATE images SET label = ? WHERE id = ?', (label, id))
    conn.commit()
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
