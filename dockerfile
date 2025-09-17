FROM python:3.12-slim

# 1️⃣ Install OS dependencies (rarely changes)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy only requirements first (rarely changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy the rest of the app (changes frequently)
COPY . .

# 5️⃣ Expose port and run Gunicorn
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
