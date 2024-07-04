# Specify the base image (replace with your desired Python version)
FROM python:3.11.4

# Create a working directory within the container
WORKDIR /app

# Copy your Python file to the working directory
COPY requirements.txt .
RUN pip install -r requirements.txt  
# Install dependencies from requirements.txt
RUN pip install git+https://github.com/speechbrain/speechbrain.git@develop

# Copy your Python file to the working directory
COPY User_interface.py .

# Set the command to execute your Python file
CMD ["python", "User_interface.py"]

#images=[app4]