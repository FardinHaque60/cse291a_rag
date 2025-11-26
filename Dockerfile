FROM python:3.12-slim

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command drops user into a shell
CMD ["bash"]
