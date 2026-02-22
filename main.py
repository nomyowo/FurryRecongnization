import os
import shutil
import sqlite3
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, status, Query
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename
import db
import zipfile
import tempfile

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
            label TEXT NOT NULL, -- Deprecated, kept for backward compatibility but data moved to image_labels
            type TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # New table for multiple labels support
    conn.execute('''
        CREATE TABLE IF NOT EXISTS image_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
        )
    ''')

    # Check if migration is needed (if image_labels is empty but images has data)
    cursor = conn.execute("SELECT COUNT(*) FROM image_labels")
    if cursor.fetchone()[0] == 0:
        print("Migrating labels to image_labels table...")
        # Get all images with non-empty labels
        images = conn.execute("SELECT id, label FROM images WHERE label != '' AND label IS NOT NULL").fetchall()
        for img in images:
            # Split by comma if multiple labels exist in old format, or just take the single label
            labels = [l.strip() for l in img['label'].split(',') if l.strip()]
            for lbl in labels:
                conn.execute("INSERT INTO image_labels (image_id, label) VALUES (?, ?)", (img['id'], lbl))
        conn.commit()
        print("Migration complete.")

    conn.commit()
    conn.close()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/lib")
def view_lib(request: Request, label: Optional[str] = Query(None), conn: sqlite3.Connection = Depends(get_db)):
    if label == "":
        label = None

    # Reverted to single label logic for lib
    # Fetch distinct labels and counts from images table directly
    db_labels = conn.execute('SELECT label, COUNT(*) as count FROM images WHERE type = ? GROUP BY label ORDER BY label', ('lib',)).fetchall()

    labels = []
    for row in db_labels:
        if row['label']:
            labels.append({'label': row['label'], 'count': row['count']})

    images = []
    if label is not None:
        images = conn.execute('SELECT * FROM images WHERE type = ? AND label = ? ORDER BY id DESC', ('lib', label)).fetchall()

    # No need to attach multiple labels logic for lib as it is single label

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

            # Single label insert for lib
            conn.execute('INSERT INTO images (filename, label, type, timestamp) VALUES (?, ?, ?, ?)',
                         (save_name, label, 'lib', str(datetime.now())))
            # No image_labels insertion for lib

    conn.commit()
    return RedirectResponse(url="/lib", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/query")
def view_query(request: Request, label: Optional[str] = Query(None), conn: sqlite3.Connection = Depends(get_db)):
    # Fetch distinct labels and counts from image_labels joined with images where type is query
    db_labels = conn.execute('''
        SELECT il.label, COUNT(DISTINCT il.image_id) as count 
        FROM image_labels il
        JOIN images i ON i.id = il.image_id 
        WHERE i.type = 'query' 
        GROUP BY il.label 
        ORDER BY il.label
    ''').fetchall()

    labels = []

    # Check for unlabeled query images
    unlabeled_count = conn.execute("SELECT COUNT(*) FROM images WHERE type = 'query' AND id NOT IN (SELECT image_id FROM image_labels)").fetchone()[0]
    if unlabeled_count > 0:
        labels.append({'label': '未检测到', 'original_label': '', 'count': unlabeled_count})

    for row in db_labels:
        labels.append({'label': row['label'], 'original_label': row['label'], 'count': row['count']})

    images = []

    # Check if we are filtering by a specific label
    if label is not None:
        if label == "未检测到":
             # Special case: fetch images with no label relations
             images = conn.execute("SELECT * FROM images WHERE type = 'query' AND id NOT IN (SELECT image_id FROM image_labels) ORDER BY id DESC").fetchall()
        else:
             images = conn.execute('''
                SELECT i.* 
                FROM images i
                JOIN image_labels il ON i.id = il.image_id
                WHERE i.type = 'query' AND il.label = ? 
                ORDER BY i.id DESC
             ''', (label,)).fetchall()

    # Attach concatenated labels to images for display
    images_with_labels = []
    for img in images:
        img_dict = dict(img)
        lbls = conn.execute("SELECT label FROM image_labels WHERE image_id = ?", (img['id'],)).fetchall()
        img_dict['label'] = ", ".join([l[0] for l in lbls])
        images_with_labels.append(img_dict)

    return templates.TemplateResponse("query.html", {"request": request, "images": images_with_labels, "labels": labels, "current_label": label})


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

            # Normalize label: replace English comma with Chinese comma for legacy storage/display consistency if desired,
            # though we split by both below.
            # The user asked "saving labels convert English comma to Chinese comma".
            # Since we store in `images.label` (legacy) and `image_labels` (normalized), let's ensure `images.label` uses Chinese comma.
            normalized_label_str = label.replace(',', '，') if label else ''

            # Insert into images
            cursor = conn.execute('INSERT INTO images (filename, label, type, timestamp) VALUES (?, ?, ?, ?)',
                         (save_name, normalized_label_str, 'query', str(datetime.now())))
            image_id = cursor.lastrowid

            # Process multiple labels
            if label:
                # Split by both English and Chinese commas
                label_list = [l.strip() for l in label.replace('，', ',').split(',') if l.strip()]
                for lbl in label_list:
                    conn.execute('INSERT INTO image_labels (image_id, label) VALUES (?, ?)', (image_id, lbl))

    conn.commit()
    return RedirectResponse(url="/query", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/update_label/{id}")
def update_label(
    id: int,
    label: str = Form(...),
    redirect_to: str = Form("/"),
    conn: sqlite3.Connection = Depends(get_db)
):
    # Check image type to decide behavior
    img = conn.execute('SELECT type FROM images WHERE id = ?', (id,)).fetchone()
    if img and img['type'] == 'lib':
        # Single label update for lib
        conn.execute('UPDATE images SET label = ? WHERE id = ?', (label, id))
    else:
        # Multi-label update for query
        # Normalize for legacy column
        normalized_label_str = label.replace(',', '，')
        conn.execute('UPDATE images SET label = ? WHERE id = ?', (normalized_label_str, id))

        # Delete existing labels for this image
        conn.execute('DELETE FROM image_labels WHERE image_id = ?', (id,))

        # Insert new labels, splitting by both comma types
        label_list = [l.strip() for l in label.replace('，', ',').split(',') if l.strip()]
        for lbl in label_list:
            conn.execute('INSERT INTO image_labels (image_id, label) VALUES (?, ?)', (id, lbl))

    conn.commit()
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)


@app.get("/query/download_all")
def download_all_query(conn: sqlite3.Connection = Depends(get_db)):
    # Create a temporary file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Get all query images
            images = conn.execute("SELECT * FROM images WHERE type = ?", ('query',)).fetchall()
            for img in images:
                file_path = os.path.join(UPLOAD_FOLDER_QUERY, img['filename'])
                if os.path.exists(file_path):
                    # Get all labels for this image
                    labels = conn.execute("SELECT label FROM image_labels WHERE image_id = ?", (img['id'],)).fetchall()
                    label_list = [l[0] for l in labels]

                    if not label_list:
                        # Unlabeled
                        zf.write(file_path, arcname=os.path.join('未检测到', img['filename']))
                    else:
                        # Add to each label folder
                        for lbl in label_list:
                            zf.write(file_path, arcname=os.path.join(lbl, img['filename']))

        return FileResponse(
            temp_zip.name,
            media_type='application/zip',
            filename=f"query_images_all_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        )
    except Exception as e:
        if os.path.exists(temp_zip.name):
            os.remove(temp_zip.name)
        raise e

@app.get("/query/download_label")
def download_query_label(label: str = Query(...), conn: sqlite3.Connection = Depends(get_db)):
    # Create a temporary file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Handle "未检测到" case
            if label == "未检测到":
                images = conn.execute("SELECT * FROM images WHERE type = ? AND id NOT IN (SELECT image_id FROM image_labels)", ('query',)).fetchall()
            else:
                images = conn.execute('''
                    SELECT i.* 
                    FROM images i
                    JOIN image_labels il ON i.id = il.image_id
                    WHERE i.type = 'query' AND il.label = ?
                ''', (label,)).fetchall()

            for img in images:
                file_path = os.path.join(UPLOAD_FOLDER_QUERY, img['filename'])
                if os.path.exists(file_path):
                    # Add file to zip at root level since it's a single label download
                    zf.write(file_path, arcname=img['filename'])

        filename_label = label if label else "unknown"
        return FileResponse(
            temp_zip.name,
            media_type='application/zip',
            filename=f"query_images_{filename_label}_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        )
    except Exception as e:
        if os.path.exists(temp_zip.name):
            os.remove(temp_zip.name)
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
