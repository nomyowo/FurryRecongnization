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


# TODO: 2、将furry图片检查加入系统
#       3、加入标签选择功能
#       4、加入标签查找
#       5、加入登录、权限功能（加入多线程）
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

def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.on_event("startup")
def startup_event():
    db.init_db()

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/lib")
def view_lib(request: Request, label: Optional[str] = Query(None)):
    if label == "":
        label = None

    # Fetch distinct labels and counts from images table directly
    labels = db.get_lib_labels_stats()

    images = []
    if label is not None:
        images = db.get_lib_images(label)

    # No need to attach multiple labels logic for lib as it is single label

    return templates.TemplateResponse("lib.html", {"request": request, "images": images, "labels": labels, "current_label": label})


@app.post("/lib/upload")
def upload_lib(
    request: Request,
    files: List[UploadFile] = File(...),
    label: str = Form(""),
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
            db.add_image(save_name, label, 'lib', str(datetime.now()))
            # No image_labels insertion for lib

    return RedirectResponse(url="/lib", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/query")
def view_query(request: Request, label: Optional[str] = Query(None)):
    # Fetch distinct labels and counts from image_labels joined with images where type is query
    labels = db.get_query_labels_stats()

    images = []

    # Check if we are filtering by a specific label
    if label is not None:
        images = db.get_query_images(label)

    return templates.TemplateResponse("query.html", {"request": request, "images": images, "labels": labels, "current_label": label})


@app.post("/query/upload")
def upload_query(
    request: Request,
    files: List[UploadFile] = File(...),
    label: Optional[str] = Form(""),
):
    for file in files:
        if file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_name = f"{timestamp}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER_QUERY, save_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Normalize label
            normalized_label_str = label.replace(',', '，') if label else ''

            # Insert into images
            image_id = db.add_image(save_name, normalized_label_str, 'query', str(datetime.now()))

            # Process multiple labels
            if label:
                # Split by both English and Chinese commas
                label_list = [l.strip() for l in label.replace('，', ',').split(',') if l.strip()]
                db.add_image_labels(image_id, label_list)

    return RedirectResponse(url="/query", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/update_label/{id}")
def update_label(
    id: int,
    label: Optional[str] = Form(None),
    redirect_to: str = Form("/"),
):
    # Check image type to decide behavior
    img_type = db.get_image_type(id)
    if img_type == 'lib':
        # Single label update for lib
        db.update_image_label_legacy(id, label)
    else:
        # Multi-label update for query
        # Normalize for legacy column
        normalized_label_str = label.replace(',', '，') if label is not None else ''

        # Split by both comma types
        label_list = [l.strip() for l in (label or '').replace('，', ',').split(',') if l.strip()]

        db.update_image_labels_query(id, normalized_label_str, label_list)

    return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)


@app.get("/query/download_all")
def download_all_query():
    # Create a temporary file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Get all query images
            images_data = db.get_all_query_images_with_labels()

            for item in images_data:
                img = item['image']
                label_list = item['labels']
                file_path = os.path.join(UPLOAD_FOLDER_QUERY, img['filename'])

                if os.path.exists(file_path):
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
def download_query_label(label: str = Query(...)):
    # Create a temporary file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            images = db.get_query_images_by_label_raw(label)

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
