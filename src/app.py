import traceback
from uuid import uuid4
from typing import cast, Union, Optional, Dict, List

import torch
from flasgger import Swagger
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector,
    NamedSparseVector,
    FieldCondition,
    MatchAny,
    Filter,
    Condition,
)

from services import LoggerService
from models import load_dense_model
from models import load_sparse_model

# from models.reranker_model import load_reranker_model

logger_service = LoggerService(
    log_level="INFO",
    log_format="%(asctime)s - %(name)s - %(session)s - %(levelname)s - %(message)s",
    log_dir="logs",
)

app = Flask(__name__)
app.url_map.strict_slashes = False

swagger = Swagger(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger_service.info(
    f"Loading dense models onto {device}...",
    extra={"session": "SYSTEM"},
)

try:
    dense_models = {
        "jinaai/jina-embeddings-v3": load_dense_model(
            "jinaai/jina-embeddings-v3", device
        )
    }
except:
    stack_trace = traceback.format_exc()
    logger_service.error(
        f"Exception during dense models loading\nStack Trace: {stack_trace}",
        extra={"session": "SYSTEM"},
    )
    raise

logger_service.info(
    f"Loading sparse models onto {device}...",
    extra={"session": "SYSTEM"},
)

try:
    sparse_models = {
        "lazydatascientist/splade-v3": load_sparse_model(
            "lazydatascientist/splade-v3", device
        )
    }
except:
    stack_trace = traceback.format_exc()
    logger_service.error(
        f"Exception during sparse models loading\nStack Trace: {stack_trace}",
        extra={"session": "SYSTEM"},
    )
    raise


# reranker_models = {
#     'bge-reranker-v2-m3': load_reranker_model('BAAI/bge-reranker-v2-m3')
# }

logger_service.info(
    f"Loading Qdrant client...",
    extra={"session": "SYSTEM"},
)

qdrant_client = QdrantClient(path="qdrant_storage")


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
        examples:
          application/json:
            embeddings: [[0.0732, -0.0805, 0.1269], [0.0349, 0.0810, -0.1337]]
      404:
        description: The specified dense model was not found.
      500:
        description: An error occurred while generating the dense embeddings.
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


@app.route("/embed/sparse", methods=["POST"])
def embed_sparse():
    """
    Generate sparse embeddings (indices and values of non-zero elements) for a list of texts.
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
              description: Name of the sparse embedding model to use.
            texts:
              type: array
              items:
                type: string
              description: List of texts to embed.
          required:
            - model_name
            - texts
    responses:
        200:
            description: A dictionary containing lists of indices and values for each input text's sparse embeddings.
            schema:
            type: object
            properties:
                indices:
                type: array
                items:
                    type: array
                    items:
                    type: integer
                description: List of lists, where each inner list contains the indices for one input text.
                values:
                type: array
                items:
                    type: array
                    items:
                    type: number
                description: List of lists, where each inner list contains the values for one input text.
            examples:
                application/json:
                    indices: [[1, 5, 23], [2, 8, 15]]
                    values: [[0.87, 0.65, 0.92], [0.78, 0.55, 0.81]]
        404:
            description: The specified sparse model was not found.
        500:
            description: An error occurred while generating the sparse embeddings.
    """
    session = str(uuid4())

    data = request.get_json()
    texts = data.get("texts", [])
    model_name = cast(str, data.get("model_name", "")).lower()

    if model_name in sparse_models:
        tokenizer, model = sparse_models[model_name]
    else:
        logger_service.error(
            f"Error: Sparse model '{model_name}' not found.",
            extra={"session": session},
        )
        return jsonify({"error": "Model not found"}), 404

    try:
        logger_service.info(
            f"Encoding {len(texts)} texts using model: {model_name}",
            extra={"session": session},
        )

        encoded_inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded_inputs).logits
            relu_logits = torch.nn.functional.relu(logits)
            batch_max_activations = torch.max(relu_logits, dim=1)[0]

        all_indices = []
        all_values = []
        for i in range(batch_max_activations.size(0)):
            max_activations = batch_max_activations[i]
            indices = max_activations.nonzero(as_tuple=True)[0].tolist()
            values = max_activations[indices].tolist()
            all_indices.append(indices)
            all_values.append(values)
        logger_service.info(
            f"Successfully encoded {len(all_indices)} sparse embeddings using model: {model_name}",
            extra={"session": session},
        )
        return jsonify({"indices": all_indices, "values": all_values})
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger_service.error(
            f"Exception during embedding with model '{model_name}': {e}\nStack Trace: {stack_trace}",
            extra={"session": session},
        )
        return jsonify({"error": "Failed to generate embeddings"}), 500


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


def __search(
    collection_name: str,
    query_embedding: Union[NamedVector, NamedSparseVector],
    must_conditions: Optional[Dict[str, List[str]]] = None,
    top_k: int = 12,
):
    search_filter = None
    if must_conditions:
        must = [
            FieldCondition(key=key, match=MatchAny(any=values))
            for key, values in must_conditions.items()
        ]
        search_filter = Filter(must=cast(List[Condition], must))
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=top_k,
    )

    return [
        {
            "id": result.id,
            "version": result.version,
            "score": result.score,
            "payload": result.payload,
            "vector": result.vector,
            "shard_key": result.shard_key,
            "order_value": result.order_value,
        }
        for result in results
    ]


@app.route("/search/dense", methods=["POST"])
def search_dense():
    """
    Search dense vectors.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            collection_name:
              type: string
              description: Name of the collection to search.
            query_vector:
              type: object
              properties:
                name:
                  type: string
                vector:
                  type: array
                  items:
                    type: number
              description: Dense vector data with name.
            must_conditions:
              type: object
              description: Optional dictionary with field filters (list of allowed values).
            top_k:
              type: integer
              description: Number of top results to return (default 12).
    responses:
      200:
        description: List of matched results with id, score, and payload.
    """
    try:
        data = request.get_json()

        collection_name = data["collection_name"]
        query_vector = data["query_vector"]
        must_conditions = data.get("must_conditions")
        top_k = data.get("top_k", 12)

        query_embedding = NamedVector(**query_vector)

        must_conditions_dict = None
        if must_conditions is not None:
            must_conditions_dict = cast(Dict[str, List[str]], must_conditions)

        results = __search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            must_conditions=must_conditions_dict,
            top_k=top_k,
        )

        return jsonify(results), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/search/sparse", methods=["POST"])
def search_sparse():
    """
    Search sparse vectors.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            collection_name:
              type: string
              description: Name of the collection to search.
            query_vector:
              type: object
              properties:
                name:
                  type: string
                vector:
                  type: object
                  properties:
                    indices:
                      type: array
                      items:
                        type: integer
                    values:
                      type: array
                      items:
                        type: number
              description: Sparse vector data with name.
            must_conditions:
              type: object
              description: Optional dictionary with field filters (list of allowed values).
            top_k:
              type: integer
              description: Number of top results to return (default 12).
    responses:
      200:
        description: List of matched results with id, score, and payload.
    """
    try:
        data = request.get_json()

        collection_name = data["collection_name"]
        query_vector = data["query_vector"]
        must_conditions = data.get("must_conditions")
        top_k = data.get("top_k", 12)

        query_embedding = NamedSparseVector(**query_vector)

        must_conditions_dict = None
        if must_conditions is not None:
            must_conditions_dict = cast(Dict[str, List[str]], must_conditions)

        results = __search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            must_conditions=must_conditions_dict,
            top_k=top_k,
        )

        return jsonify(results), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
