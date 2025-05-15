# Digit Classification 

A FastAPI-based microservice for digit classification, designed following the [12-Factor App](https://12factor.net/) methodology to promote best practices in configuration, scalability, and deployment.

The digit classifier model is trained on the MNIST dataset using techniques from [Machine Learning with PyTorch and Scikit-Learn](https://sebastianraschka.com/blog/2022/ml-pytorch-book.html) by Sebastian Raschka, Yuxi Liu, and Vahid Mirjalili. The classifier expects **grayscale images of handwritten digits, ideally 28x28 pixels in size**, similar to those in the MNIST dataset.

## How to Run the Application

### Running Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/yyohyy/12-Factor-Digit-Classifier.git
   cd 12-Factor-Digit-Classifier
   ```
2. Create and activate a Python virtual environment[on Windows]:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the FastAPI server:
    ```bash
    uvicorn app.main:app --reload
    ```
5. Access the API docs:
   
   Open your browser and go to http://localhost:8000/docs to predict digit in an image.

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t digit-classifier .
   ```
2. Run the Docker container:
    ```bash
    docker run -d -p 8000:8000 digit-classifier
    ```
3. Install dependencies:
   
   Open your browser at http://localhost:8000/docs to predict digit in an image.

### Configuration

.env.example  

```
   MODEL_PATH=path/to/your/model  
   LOG_PATH=logs/
```

### Testing
Run tests locally with:
```
pytest tests/
```
