# Base image
FROM ubuntu:20.04

# Set environment variable to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9, pip, and OpenJDK 11
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Upgrade pip and set python3.9 as default Python
RUN python3.9 -m pip install --upgrade pip && ln -sf /usr/bin/python3.9 /usr/bin/python

# Working directory
WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the application port
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
