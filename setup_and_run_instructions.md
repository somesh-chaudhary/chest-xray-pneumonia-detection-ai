# Setup and Run Instructions

Follow these simple steps to run the PneumoDetect AI app locally with the pre-trained model.

## 1. Prerequisites
You need **Python 3.8 to 3.11** installed. The project uses `tensorflow==2.13.0` which is compatible with these Python versions.

## 2. Open Terminal
Open a terminal (e.g., PowerShell or Command Prompt) and navigate to the project's root folder:
```powershell
cd C:\Users\somes\Downloads\chest-xray-pneumonia-detection-ai-main
```

## 3. Install Dependencies
Install the required packages for the Streamlit app. Note that we use the `requirements.txt` specifically from the `streamlit_api_folder` which contains `streamlit`, `tensorflow`, and other UI/processing libraries.

```powershell
pip install -r api/streamlit_api_folder/requirements.txt
```

## 4. Run the Streamlit Application
Start the Streamlit UI. (FastAPI is not required for the Streamlit app to function, as the model is loaded and run directly by the Streamlit application.)

```powershell
streamlit run api/streamlit_api_folder/streamlit_app.py
```

## 5. View in Browser
Once Streamlit starts, it will provide a local URL. Open the following link in your browser if it doesn't open automatically:
**http://localhost:8501**

---

### Optional: Run FastAPI Backend
If you want to try the REST API separately, you can run it with the following commands:
```powershell
cd C:\Users\somes\Downloads\chest-xray-pneumonia-detection-ai-main\api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Access the API docs at: **http://localhost:8000/docs**
