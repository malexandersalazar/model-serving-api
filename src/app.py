from flasgger import Swagger
from flask import Flask, request, jsonify

from models.dense_model import load_dense_model

# from models.sparse_model import load_sparse_model
# from models.reranker_model import load_reranker_model

app = Flask(__name__)
swagger = Swagger(app)

# Load models onto GPU at startup
dense_models = {"jinaai/jina-embeddings-v3": load_dense_model("jinaai/jina-embeddings-v3")}

# sparse_models = {
#     'splade-v3': load_sparse_model('naver/splade-v3')
# }

# reranker_models = {
#     'bge-reranker-v2-m3': load_reranker_model('BAAI/bge-reranker-v2-m3')
# }


@app.route("/embed/dense/<path:model_name>", methods=["POST"])
def embed_dense(model_name):
    """
    Generate dense embeddings for a list of texts.
    ---
    parameters:
      - name: model_name
        in: path
        type: string
        required: true
        description: Name of the dense embedding model.
      - name: texts
        in: body
        required: true
        schema:
          type: object
          properties:
            texts:
              type: array
              items:
                type: string
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
    data = request.get_json()
    texts = data.get("texts", [])
    model = dense_models.get(model_name)
    if not model:
        return jsonify({"error": "Model not found"}), 404
    embeddings = model.encode(texts, convert_to_tensor=True).tolist()
    return jsonify({"embeddings": embeddings})


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
