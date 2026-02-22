import os
import shutil
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, status, Query
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename
import db
from classifier import FurryClassifier
import zipfile
import tempfile
import base64
from io import BytesIO
from PIL import Image
import numpy as np



# TODO: 3、加入标签选择功能
#       4、加入标签查找
#       5、加入登录、权限功能（加入多线程）
app = FastAPI()

# 分类器
classifier: Optional[FurryClassifier] = None

# 文件上传配置
UPLOAD_FOLDER_LIB = 'lib'
UPLOAD_FOLDER_QUERY = 'query'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER_LIB, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_QUERY, exist_ok=True)

# 挂载静态文件目录
app.mount("/files/lib", StaticFiles(directory=UPLOAD_FOLDER_LIB), name="lib_files")
app.mount("/files/query", StaticFiles(directory=UPLOAD_FOLDER_QUERY), name="query_files")

templates = Jinja2Templates(directory="templates")

# 文件类型检查
def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 初始化分类器
def init_classifier():
    global classifier
    try:
        lib_imgs = db.get_all_lib_images()
        # Convert to list of (label, path)
        lib_data = []
        for img in lib_imgs:
            path = Path(UPLOAD_FOLDER_LIB) / img['filename']
            if path.exists():
                lib_data.append((img['label'], path))

        if lib_data:
            print(f"Initializing classifier with {len(lib_data)} images...")
            classifier = FurryClassifier(lib_data, backend="yolo")
        else:
            print("No library images found. Classifier not initialized.")
            classifier = None
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        classifier = None

# 应用启动事件，初始化数据库和分类器
@app.on_event("startup")
def startup_event():
    db.init_db()
    init_classifier()

# 首页
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 库管理
@app.get("/lib")
def view_lib(request: Request, label: Optional[str] = Query(None)):
    if label == "":
        label = None

    # 获取所有标签和对应的图片数量
    labels = db.get_lib_labels_stats()

    images = []
    if label is not None:
        images = db.get_lib_images(label)

    return templates.TemplateResponse("lib.html", {"request": request, "images": images, "labels": labels, "current_label": label})


# 上传库图片
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

            # 单标签库直接使用label字段，不进行多标签处理
            db.add_image(save_name, label, 'lib', str(datetime.now()))

    # 重学习分类器
    init_classifier()

    return RedirectResponse(url="/lib", status_code=status.HTTP_303_SEE_OTHER)


# 删除库图片
@app.post("/lib/delete_bulk")
def delete_lib_images_bulk(
    image_ids: List[int] = Form(...),
    redirect_to: str = Form("/lib")
):
    for img_id in image_ids:
        filename = db.delete_image(img_id)
        if filename:
            file_path = os.path.join(UPLOAD_FOLDER_LIB, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    # 重学习分类器
    init_classifier()

    return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)


# 删除库文件夹（标签）
@app.post("/lib/delete_folders_bulk")
def delete_lib_folders_bulk(
    labels: List[str] = Form(...)
):
    for label in labels:
        filenames = db.delete_lib_folder(label)
        for filename in filenames:
            file_path = os.path.join(UPLOAD_FOLDER_LIB, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    # 重学习分类器
    init_classifier()

    return RedirectResponse(url="/lib", status_code=status.HTTP_303_SEE_OTHER)


# 标签推理图管理
@app.get("/query")
def view_query(request: Request, label: Optional[str] = Query(None)):
    # Fetch distinct labels and counts from image_labels joined with images where type is query
    labels = db.get_query_labels_stats()

    images = []

    # 如果 label 是空字符串，则视为未选择标签，显示所有图片
    if label is not None:
        images = db.get_query_images(label)

    return templates.TemplateResponse("query.html", {"request": request, "images": images, "labels": labels, "current_label": label})


# 上传推理图片
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

            # 标准化标签
            normalized_label_str = label.replace(',', '，') if label else ''

            # 插入数据库，获取新图片ID
            image_id = db.add_image(save_name, normalized_label_str, 'query', str(datetime.now()))

            # 处理多标签，存储到 image_labels 表
            if label:
                # 中英文分割符处理
                label_list = [l.strip() for l in label.replace('，', ',').split(',') if l.strip()]
                db.add_image_labels(image_id, label_list)

    return RedirectResponse(url="/query", status_code=status.HTTP_303_SEE_OTHER)


# 更新推理图片标签
@app.post("/update_label/{id}")
def update_label(
    id: int,
    label: Optional[str] = Form(None),
    redirect_to: str = Form("/"),
):
    # 根据图片ID获取类型，决定是单标签更新还是多标签更新
    img_type = db.get_image_type(id)
    if img_type == 'lib':
        # 单标签更新，直接更新 images 表的 label 字段
        db.update_image_label_legacy(id, label)
    else:
        # 多标签更新，更新 image_labels 表，并同步更新 images 表的 label 字段（逗号分隔字符串）
        # 更新分隔符
        normalized_label_str = label.replace(',', '，') if label is not None else ''

        # 处理多标签，存储到 image_labels 表
        label_list = [l.strip() for l in (label or '').replace('，', ',').split(',') if l.strip()]

        db.update_image_labels_query(id, normalized_label_str, label_list)

    return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)


# 标签推理检测
@app.post("/query/detect")
def detect_query_images(
    request: Request,
    image_ids: List[int] = Form(...),
):
    global classifier

    if not classifier:
        # Try to initialize if not ready (e.g. if startup failed or cleared)
        init_classifier()

    if not classifier:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Classifier not initialized. Please ensure there are images in the library."})

    # 结果列表
    results = []

    for img_id in image_ids:
        row = db.get_image_by_id(img_id)
        if not row:
            continue

        filename = row['filename']
        file_path = Path(UPLOAD_FOLDER_QUERY) / filename
        if not file_path.exists():
            continue

        # 进行预测
        try:
            preds = classifier.predict(file_path, topk=5)
        except Exception as e:
            print(f"Prediction failed for {filename}: {e}")
            continue

        # 构建检测结果
        detections = []
        image_array = np.array(Image.open(file_path).convert("RGB"))

        for i, p in enumerate(preds):
            # 每个预测结果包含一个_mask字段，表示检测到的区域掩码
            mask = p.get("_mask")
            if mask is not None:
                # 使用 classifier.py 中的 crop_with_mask 函数裁剪出检测区域的图像
                from classifier import crop_with_mask, to_uint8_mask
                cropped, _ = crop_with_mask(image_array, mask)

                # 将裁剪后的图像转换为 base64 编码的字符串，方便在前端直接显示
                pil_img = Image.fromarray(cropped)
                buff = BytesIO()
                pil_img.save(buff, format="PNG")
                img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

                detections.append({
                    "id": i,
                    "crop_b64": img_str,
                    "predictions": p["predictions"], # list of {name, score}
                    "top_label": p["predictions"][0]["name"] if p["predictions"] else ""
                })

        if detections:
            results.append({
                "image_id": img_id,
                "filename": filename,
                "detections": detections
            })

    return templates.TemplateResponse("review.html", {"request": request, "results": results})


# 保存用户审核的标签结果
@app.post("/query/save_results")
def save_query_results(
    request: Request,
    results: str = Form(...), # JSON string of results
):
    import json
    data = json.loads(results)

    for item in data:
        image_id = item.get("image_id")
        labels = item.get("labels", [])

        if not image_id or not labels:
            continue

        # 将标签列表转换为逗号分隔字符串存储在 images 表的 label 字段中
        label_str = ",".join(labels)

        # 更新数据库中的标签信息
        db.update_image_labels_query(image_id, label_str.replace(',', '，'), labels)


    return RedirectResponse(url="/query", status_code=status.HTTP_303_SEE_OTHER)


# 下载推理图片（按标签分类）
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


# 下载推理图片（单标签）
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