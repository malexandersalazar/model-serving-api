import traceback
from uuid import uuid4
from typing import cast
from flasgger import Swagger
from flask import Flask, request, jsonify

from services import LoggerService
from models import load_dense_model

# from models.sparse_model import load_sparse_model
# from models.reranker_model import load_reranker_model

logger_service = LoggerService(
    log_level="INFO",
    log_format="%(asctime)s - %(name)s - %(session)s - %(levelname)s - %(message)s",
    log_dir="logs",
)

app = Flask(__name__)
app.url_map.strict_slashes = False

swagger = Swagger(app)

logger_service.info(
    "Loading dense models onto GPU...",
    extra={"session": "SYSTEM"},
)
dense_models = {
    "jinaai/jina-embeddings-v3": load_dense_model("jinaai/jina-embeddings-v3")
}

# sparse_models = {
#     'splade-v3': load_sparse_model('naver/splade-v3')
# }

# reranker_models = {
#     'bge-reranker-v2-m3': load_reranker_model('BAAI/bge-reranker-v2-m3')
# }


@app.route("/embed/dense", methods=["POST"])
def embed_dense():
    """
    Generate dense embeddings for a list of texts.
    ---
    parameters:
          - name: body
            in: body
            required: true
            schema:
              type: object
              properties:
                model_name:
                  type: string
                  description: Name of the dense embedding model
                texts:
                  type: array
                  items:
                    type: string
              required:
                - model_name
                - texts
    responses:
      200:
        description: A list of dense embeddings.
        schema:
          type: object
          properties:
            embeddings:
              type: array
              items:
                type: array
                items:
                  type: number
      404:
        description: Model not found.
    """
    session = str(uuid4())

    data = request.get_json()
    texts = data.get("texts", [])
    model_name = cast(str, data.get("model_name", "")).lower()

    logger_service.info(
        f'Processing request for dense embeddings using model "{model_name}" for {len(texts)} texts.',
        extra={"session": session},
    )

    model = dense_models.get(model_name)
    if not model:
        logger_service.error(
            f"Error: Dense model '{model_name}' not found.",
            extra={"session": session},
        )
        return jsonify({"error": "Model not found"}), 404

    try:
        logger_service.info(
            f"Encoding {len(texts)} texts using model: {model_name}",
            extra={"session": session},
        )
        embeddings = model.encode(texts, convert_to_tensor=True).tolist()
        logger_service.info(
            f"Successfully encoded {len(embeddings)} dense embeddings using model: {model_name}",
            extra={"session": session},
        )
        return jsonify({"embeddings": embeddings})
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger_service.error(
            f"Exception during embedding with model '{model_name}': {e}\nStack Trace: {stack_trace}",
            extra={"session": session},
        )
        return jsonify({"error": "Failed to generate embeddings"}), 500


# @app.route('/embed/sparse/<model_name>', methods=['POST'])
# def embed_sparse(model_name):
#     data = request.get_json()
#     texts = data.get('texts', [])
#     model = sparse_models.get(model_name)
#     if not model:
#         return jsonify({'error': 'Model not found'}), 404
#     embeddings = model.encode(texts, convert_to_tensor=True).tolist()
#     return jsonify({'embeddings': embeddings})

# @app.route('/rerank/<model_name>', methods=['POST'])
# def rerank(model_name):
#     data = request.get_json()
#     query = data.get('query', '')
#     candidates = data.get('candidates', [])
#     model = reranker_models.get(model_name)
#     if not model:
#         return jsonify({'error': 'Model not found'}), 404
#     pairs = [[query, candidate] for candidate in candidates]
#     scores = model.predict(pairs).tolist()
#     return jsonify({'scores': scores})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
