from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

app = FastAPI()

data = {
    "PatientID": [1, 2, 3],
    "Age": [34, 45, 56],
    "HeartRate": [80, 120, 95],
    "BloodPressure": ["120/80", "150/90", "140/85"],
    "Cholesterol": [180, 250, 210],
    "Notes": [
        "Patient is healthy with stable vitals.",
        "High blood pressure and cholesterol detected.",
        "Mild hypertension, monitor vitals regularly."
    ],
}
df = pd.DataFrame(data)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
df['Combined'] = df['Age'].astype(str) + " " + df['HeartRate'].astype(str) + " " + df['Notes']
embeddings = embedder.encode(df['Combined'].tolist())
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

chat_model = pipeline("text-generation", model="distilgpt2", device="cpu")

class QueryRequest(BaseModel):
    query: str

class SummaryResponse(BaseModel):
    retrieved_data: dict
    summary: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Root endpoint for the API with tabular information.
    """
    html_content = """
    <html>
        <head>
            <title>Patient Summary API</title>
        </head>
        <body>
            <h1>Welcome to the Patient Summary API</h1>
            <p>This API processes patient data and generates summaries based on provided health conditions.</p>
            <h2>How to Use:</h2>
            <table border="1">
                <tr>
                    <th>Step</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>Visit <a href="/docs">/docs</a> to explore and test the API using the Swagger UI.</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td>Use <code>POST /generate_summary/</code> to submit a query (e.g., "high blood pressure") and get a summary.</td>
                </tr>
            </table>
            <h2>Example:</h2>
            <table border="1">
                <tr>
                    <th>Query</th>
                    <td>high blood pressure and cholesterol</td>
                </tr>
                <tr>
                    <th>Response</th>
                    <td>
                        <b>Retrieved Data:</b> PatientID: 2, Age: 45, HeartRate: 120, BloodPressure: 150/90, Cholesterol: 250, Notes: High blood pressure and cholesterol detected.<br>
                        <b>Summary:</b> The patient is a 45-year-old with high blood pressure and elevated cholesterol. Consider medical intervention or lifestyle changes.
                    </td>
                </tr>
            </table>
            <h2>Note:</h2>
            <p>Ensure you provide accurate health-related queries to get meaningful results.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate_summary/", response_model=SummaryResponse)
def generate_summary(query: QueryRequest):
    query_embedding = embedder.encode([query.query])
    distances, indices = index.search(query_embedding, 1)
    retrieved_data = df.iloc[indices[0]].to_dict()

    prompt = f"Patient Data: {retrieved_data['Combined']}\nSummarize this data in a layperson-friendly format."
    response = chat_model(prompt, max_length=200, do_sample=True, top_p=0.9, temperature=0.7)
    summary = response[0]['generated_text']

    return SummaryResponse(retrieved_data=retrieved_data, summary=summary)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
