# 使用官方 Python 基础镜像
FROM python:3.13-bookworm

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .

#RUN apt-get update && apt-get install -y \
#    libglx-mesa0 \
#    libglib2.0-0 \
#    libsm6 \
#    libxrender1 \
#    libxext6 \
#    && rm -rf /var/lib/apt/lists/*



RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list \
 && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list \
 && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list \
&& apt-get update \
 && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 设置环境变量（可选）
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "main.py"]
