import copy
import json
import logging
import operator
from copy import deepcopy
from datetime import datetime
from functools import reduce
from itertools import islice
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Union

import numpy as np
from tqdm import tqdm

from haystack.document_stores import BaseDocumentStore
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.errors import DuplicateDocumentError, AstraDocumentStoreError
from haystack.nodes.retriever import DenseRetriever
from haystack.schema import LABEL_DATETIME_FORMAT, Answer, Document, FilterType, Label, Span
from haystack.utils.batching import get_batches_from_generator

logger = logging.getLogger(__name__)

TYPE_METADATA_FIELD = "doc_type"
DOCUMENT_WITH_EMBEDDING = "vector"
DOCUMENT_WITHOUT_EMBEDDING = "no-vector"
LABEL = "label"

AND_OPERATOR = "$and"
IN_OPERATOR = "$in"
EQ_OPERATOR = "$eq"

DEFAULT_BATCH_SIZE = 128

# PINECONE_STARTER_POD = "starter"

DocTypeMetadata = Literal["vector", "no-vector", "label"]


def _sanitize_index(index: Optional[str]) -> Optional[str]:
    if index:
        return index.replace("_", "-").lower()
    return None


def _get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def _set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    _get_by_path(root, items[:-1])[items[-1]] = value


class AstraDocumentStore(BaseDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR. This is a hosted document store,
    this means that your vectors will not be stored locally but in the cloud. This means that the similarity
    search will be run on the cloud as well.

    It implements the Astra vector database ([https://astra.datastax.com](https://astra.datastax.com))
    to perform similarity search on vectors. In order to use this document store, you need an API key that you can
    obtain by creating an account on the [Astra website](https://astra.datastax.com).

    The document text is stored using the SQLDocumentStore, while
    the vector embeddings and metadata (for filtering) are indexed in an Astra collection.
    """

    top_k_limit = 10_000
    top_k_limit_vectors = 1_000

    def __init__(
        self,
        astra_id: str,
        astra_application_token: str,
        astra_region: str = "us-east1",
        collection_name: Optional[str] = "haystack",
        astra_keyspace: Optional[str] = None,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "cosine",
        replicas: int = 1,
        shards: int = 1,
        namespace: Optional[str] = None,
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_collection: bool = False,
        # metadata_config: Optional[Dict] = None, ## check if this is needed
        validate_index_sync: bool = True,
    ):
        """
        :param astra_id: Astra database ID ([https://astra.datastax.com/](https://astra.datastax.com/)).
        :param astra_application_token: Astra database token used to access the database via it's json API.
        :param astra_region: Astra database region defaulted to us-east1.
        :param collection_name: Astra Collection name, or also the index name
        :param astra_keyspace: Astra keyspace, under which the database/collection/index is created/will be created
        :param embedding_dim: The embedding vector size.
        :param return_embedding: Whether to return document embeddings.
        :param index: Name of index in document store to use.
        :param similarity: The similarity function used to compare document vectors. `"cosine"` is the default
            and is recommended if you are using a Sentence-Transformer model. `"dot_product"` is more performant
            with DPR embeddings.
            In both cases, the returned values in Document.score are normalized to be in range [0,1]:
                - For `"dot_product"`: `expit(np.asarray(raw_score / 100))`
                - For `"cosine"`: `(raw_score + 1) / 2`
        :param replicas: The number of replicas. Replicas duplicate the index. They provide higher availability and
            throughput.
        :param shards: The number of shards to be used in the index. We recommend to use 1 shard per 1GB of data.
        :param namespace: Optional namespace. If not specified, None is default.
        :param embedding_field: Name of field containing an embedding vector.
        :param progress_bar: Whether to show a tqdm progress bar or not.
            Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents: Handle duplicate documents based on parameter options.\
            Parameter options:
                - `"skip"`: Ignore the duplicate documents.
                - `"overwrite"`: Update any existing documents with the same ID when adding documents.
                - `"fail"`: An error is raised if the document ID of the document being added already exists.
        :param recreate_collection: If set to True, an existing Astra collection will be deleted and a new one will be
            created using the config you are using for initialization. Be aware that all data in the old collection will be
            lost if you choose to recreate the collection. Be aware that both the document_index and the label_index will
            be recreated.
        ##### TODO: check if this is needed?
        :param metadata_config: Which metadata fields should be indexed, part of the
            [selective metadata filtering](https://www.pinecone.io/docs/manage-indexes/#selective-metadata-indexing) feature.
            Should be in the format `{"indexed": ["metadata-field-1", "metadata-field-2", "metadata-field-n"]}`. By default,
            no fields are indexed.
        """

        # if metadata_config is None:
        #     metadata_config = {"indexed": []}
        # Connect to Pinecone server using python client binding
        if not astra_id:
            raise AstraDocumentStoreError(
                "Astra requires a database id, please provide one. https://astra.datastax.com"
            )

        if not astra_application_token:
            raise AstraDocumentStoreError(
                "Astra requires a database token, please provide one. https://astra.datastax.com"
            )

        if not collection_name:
            raise AstraDocumentStoreError(
                "Astra requires a collection/table name, please provide one. https://astra.datastax.com"
            )

        if not astra_keyspace:
            raise AstraDocumentStoreError(
                "Astra requires a keyspace name, please provide one. https://astra.datastax.com"
            )
        self.request_url = f"https://{astra_id}-{astra_region}.apps.astra.datastax.com/api/json/v1/{astra_keyspace}/{collection_name}"
        self.request_header = {
            "x-cassandra-token": astra_application_token,
            "Content-Type": "application/json",
        }


        ## TODO: astra initialization here
        # pinecone.init(api_key=api_key, environment=environment)
        self._astra_id = astra_id

        # Formal similarity string
        self._set_similarity_metric(similarity)

        self.similarity = similarity
        self.index: str = self._index(index)
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        # Pinecone index params
        self.replicas = replicas
        self.shards = shards
        self.namespace = namespace

        # Add necessary metadata fields to metadata_config
        fields = ["label-id", "query", TYPE_METADATA_FIELD]
        # metadata_config["indexed"] += fields
        # self.metadata_config = metadata_config

        # Initialize dictionary of index connections
        # self.pinecone_indexes: Dict[str, pinecone.Index] = {}
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field

        # Initialize dictionary to store temporary set of document IDs
        self.all_ids: dict = {}

        # Dummy query to be used during searches
        self.dummy_query = [0.0] * self.embedding_dim

        super().__init__()

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
        type_metadata: Optional[DocTypeMetadata] = None,
    ) -> int:
        pass

    def get_embedding_count(
        self, filters: Optional[FilterType] = None, index: Optional[str] = None, namespace: Optional[str] = None
    ) -> int:
        pass

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        labels: Optional[bool] = False,
        namespace: Optional[str] = None,
    ):
        pass


    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[FilterType] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        namespace: Optional[str] = None,
    ):
        pass

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
        type_metadata: Optional[DocTypeMetadata] = None,
        namespace: Optional[str] = None,
    ) -> List[Document]:
        pass

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
        type_metadata: Optional[DocTypeMetadata] = None,
        include_type_metadata: Optional[bool] = False,
    ) -> Generator[Document, None, None]:

       pass


    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
        namespace: Optional[str] = None,
        include_type_metadata: Optional[bool] = False,
    ) -> List[Document]:
        """
        Retrieves all documents in the index using their IDs.

        :param ids: List of IDs to retrieve.
        :param index: Optional index name to retrieve all documents from.
        :param batch_size: Number of documents to retrieve at a time. When working with large number of documents,
            batching can help reduce memory footprint.
        :param headers: Pinecone does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        :param namespace: Optional namespace to retrieve document from. If not specified, None is default.
        :param include_type_metadata: Indicates if `doc_type` value will be included in document metadata or not.
            If not specified, `doc_type` field will be dropped from document metadata.
        """
        pass

    def get_document_by_id(
        self,
        id: str,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
        namespace: Optional[str] = None,
    ) -> Document:
        """
        Returns a single Document retrieved using an ID.

        :param id: ID string to retrieve.
        :param index: Optional index name to retrieve all documents from.
        :param headers: Pinecone does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        :param namespace: Optional namespace to retrieve document from. If not specified, None is default.
        """
        pass

    def update_document_meta(self, id: str, meta: Dict[str, str], index: Optional[str] = None):
        """
        Update the metadata dictionary of a document by specifying its string ID.

        :param id: ID of the Document to update.
        :param meta: Dictionary of new metadata.
        :param namespace: Optional namespace to update documents from.
        :param index: Optional index name to update documents from.
        """
        pass

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        drop_ids: Optional[bool] = True,
        namespace: Optional[str] = None,
        type_metadata: Optional[DocTypeMetadata] = None,
    ):
        """
        Delete documents from the document store.
        """
        pass



    def delete_index(self, index: Optional[str]):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        pass

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
        namespace: Optional[str] = None,
        type_metadata: Optional[DocTypeMetadata] = None,
    ) -> List[Document]:

        return_documents = []
        return return_documents


    @classmethod
    def load(cls):
        """
        Default class method used for loading indexes. Not applicable to PineconeDocumentStore.
        """
        raise NotImplementedError("load method not supported for PineconeDocumentStore")

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        namespace: Optional[str] = None,
    ):
        pass

    def get_all_labels(
        self,
        index=None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
    ):
        """
        Default class method used for getting all labels.
        """
        pass

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Default class method used for counting labels. Not supported by PineconeDocumentStore.
        """
        raise NotImplementedError("Labels are not supported by PineconeDocumentStore.")

    def write_labels(
        self, labels, index=None, headers: Optional[Dict[str, str]] = None, namespace: Optional[str] = None
    ):
        """
        Default class method used for writing labels.
        """
        pass
