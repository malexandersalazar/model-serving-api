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

## ⚡ Quickstart

### 1. Install Dependencies

```bash
git clone https://github.com/your-org/model-serving-api.git
cd model-serving-api
pip install -r requirements.txt
```

### 2. Configure Your Models

Inside `app.py` you can configure:
- Which Dense model to load (e.g., `jinaai/jina-embeddings-v3`)

Models are loaded at startup into GPU memory.

Example in `app.py`:

```python
dense_model = SentenceTransformer('jinaai/jina-embeddings-v3', device='cuda')
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

| Method | URL | Description |
|:------|:----|:------------|
| `POST` | `/embed/dense/{model_name}` | Get dense embeddings for a list of texts |

All POST endpoints accept JSON payloads.

### Examples:

**Dense Embedding Request:**
```json
POST /embed/dense/jinaai%2Fjina-embeddings-v3
{
  "texts": ["Hello world", "Deep learning is awesome"]
}
```

## ⚙️ Architecture

- **Flask** for minimal REST API.
- **Sentence Transformers (Hugging Face)** for model loading and inference.
- **GPU support** via PyTorch or Tensorflow (automatically detected).
- **Swagger/OpenAPI** via `flasgger`.

## 💬 How to Contribute

We love contributions from the community!

- Submit issues for bugs, questions, and feature requests.
- Open pull requests with improvements.
- Help improve the documentation.
- Suggest new model families to support.

> See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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
- [ ] Add ONNX export support for faster model serving.
- [ ] Docker-ready containerization scripts.
- [ ] Automatic model hot-swapping without restart.

# 🏛️ Empowering AI Independence

> "Building open, accessible, sovereign AI infrastructure — one model at a time."