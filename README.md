# 🧠 Model Serving API

**A lightweight, production-ready API to serve Dense Embeddings, Sparse Embeddings, and Re-ranking models using your own hardware.**  
Built for independent developers, universities, and research institutes who want full control of their AI pipelines without relying on third-party cloud services.

## 🌎 Why This Project?

Many AI services today depend heavily on centralized infrastructure.
  
**Model Serving API** empowers you to **deploy state-of-the-art language models** — locally, on your own servers or edge devices — **cost-effectively** and **without vendor lock-in**.

- 🖥️ Bring your models to your own GPU or local cluster.
- 🛡️ Stay independent from expensive third-party APIs.
- 🚀 Serve industry-standard Hugging Face models through a simple, fast REST API.
- 🏛️ Ideal for education, research, and independent innovation.

## ✨ Features

- **Serve Dense Embeddings** (for semantic search, vector databases, etc.)
- **GPU acceleration** (leveraging PyTorch, TensorFlow backend via Hugging Face)
- **Swagger (OpenAPI)** automatic documentation for easy exploration.
- **Fully documented and extensible** for custom model provisioning.

## 🚀 Quickstart

### 1. Full Setup Instructions

#### Clone the Repository

```bash
git clone https://github.com/malexandersalazar/model-serving-api.git
cd model-serving-api/src
```

#### Create and Activate Virtual Environment

- **On Windows**:

```bash
python -m venv .venv
.venv\Scripts\activate
```

- **On Linux/macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install the Dependencies

```bash
pip install -r requirements.txt
```

#### Execute the Flask Application

```bash
python app.py
```

### 2. Configure Your Models

Inside `app.py` you can configure which dense, sparse or reranker models have to be loaded at startup into GPU memory:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dense_models = {
    "jinaai/jina-embeddings-v3": load_dense_model("jinaai/jina-embeddings-v3", device)
}

sparse_models = {
    "naver/splade-v3": load_sparse_model("naver/splade-v3", device)
}

reranker_models = {
    "baai/bge-reranker-v2-m3": load_reranker_model("BAAI/bge-reranker-v2-m3", device)
}
```

> 🔥 You can swap models easily — just change the model names!

### 3. Run the API

```bash
python app.py
```

Your API will be running on:

```
http://127.0.0.1:5000
```

Visit:

```
http://127.0.0.1:5000/apidocs/
```
to view the **interactive Swagger UI** documentation! 🎉

## 📚 API Endpoints

This API provides endpoints for generating text embeddings (both dense and sparse) and for reranking a list of candidate texts based on their relevance to a given query. All endpoints accept JSON payloads in the request body and return JSON responses.

| Method | URL            | Description                                                                     |
| :----- | :------------- | :------------------------------------------------------------------------------ |
| `POST` | `/embed/sparse` | Generates sparse embeddings (indices and values) for a list of input texts.    |
| `POST` | `/embed/dense`  | Generates dense embeddings (vector representations) for a list of input texts. |
| `POST` | `/rerank`      | Reranks a list of candidate texts based on their relevance to a query.          |

### `/embed/sparse`

This endpoint generates sparse embeddings for a list of input texts using a specified model. Sparse embeddings are represented by the indices and corresponding non-zero values in a high-dimensional space.

**Request Body Example:**

```json
POST /embed/sparse
{
  "model_name": "lazydatascientist/splade-v3",
  "texts": [
    "Natural language processing techniques.",
    "Computer vision and image recognition."
  ]
}
```

**Response Body Example:**
```json
{
  "indices": [
    [12, 45, 103, 567],
    [34, 78, 212, 890, 951]
  ],
  "values": [
    [0.76, 0.52, 0.89, 0.61],
    [0.81, 0.69, 0.72, 0.58, 0.93]
  ]
}
```

### `/embed/dense`

This endpoint generates dense embeddings for a list of input texts using a specified model. Dense embeddings are vector representations where each dimension typically holds a floating-point value.

**Request Body Example:**

```json
POST /embed/dense
{
  "model_name": "jinaai/jina-embeddings-v3",
  "texts": [
    "The future of renewable energy.",
    "Exploring the cosmos with new telescopes."
  ]
}
```

**Response Body Example:**
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, 0.987, -0.654, ...],
    [0.987, 0.654, -0.321, 0.012, -0.789, ...]
  ]
}
```

### `/rerank`

This endpoint takes a query and a list of candidate texts and returns the candidates reranked according to their relevance to the query, along with their corresponding relevance scores.

**Request Body Example:**

```json
POST /rerank
{
  "model_name": "baai/bge-reranker-v2-m3",
  "query": "Artificial intelligence applications in healthcare",
  "candidates": [
    "AI-powered diagnostic tools for early disease detection.",
    "The impact of quantum computing on medical research.",
    "Machine learning algorithms for personalized treatment plans.",
    "Telemedicine platforms and remote patient monitoring.",
    "Ethical considerations in using big data for healthcare analytics."
  ]
}
```

**Response Body Example:**
```json
{
  "candidates": [
    "AI-powered diagnostic tools for early disease detection.",
    "Machine learning algorithms for personalized treatment plans.",
    "Ethical considerations in using big data for healthcare analytics.",
    "Telemedicine platforms and remote patient monitoring.",
    "The impact of quantum computing on medical research."
  ],
  "scores": [
    0.95,
    0.91,
    0.82,
    0.78,
    0.65
  ]
}
```

## ⚙️ Architecture

- **Flask** for minimal REST API.
- **Sentence Transformers (Hugging Face)** for model loading and inference.
- **GPU support** via PyTorch or Tensorflow (automatically detected).
- **Swagger/OpenAPI** via `flasgger`.

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE).

> Free to use for commercial and academic purposes — no restrictions!

## ❤️ Acknowledgments

- [Hugging Face 🤗](https://huggingface.co) for making open model access possible.
- [Sentence Transformers](https://www.sbert.net) for incredible embedding models.
- Everyone contributing to decentralized AI development!

## 🚀 Roadmap (Planned)

- [ ] Add batch inference support for massive throughput.
- [ ] Add authentication / API keys.
- [ ] Docker-ready containerization scripts.
- [ ] Automatic model hot-swapping without restart.

# 🏛️ Empowering AI Independence

> "Building open, accessible, sovereign AI infrastructure — one model at a time."