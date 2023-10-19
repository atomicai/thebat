import abc
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Generator

from thebat.etc import Document, DocumentStoreError, DuplicateDocumentError, Label
from thebat.etc.schema import FilterType
from thebat.etc.filter import LogicalFilterClause
import time

logger = logging.getLogger(__name__)


class IDocStore(abc.ABC):
    duplicate_documents_options: tuple = ("skip", "overwrite", "fail")
    ids_iterator = None

    @abc.abstractclassmethod
    def write_documents(self, docs, index: str = None):
        pass

    @abc.abstractmethod
    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        pass

    @abc.abstractmethod
    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                        __Example__:

                        ```python
                        filters = {
                            "$and": {
                                "type": {"$eq": "article"},
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": {"$in": ["economy", "politics"]},
                                    "publisher": {"$eq": "nytimes"}
                                }
                            }
                        }
                        ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        pass

    def __iter__(self):
        if not self.ids_iterator:
            self.ids_iterator = [x.id for x in self.get_all_documents()]
        return self

    def __next__(self):
        if len(self.ids_iterator) == 0:
            raise StopIteration
        curr_id = self.ids_iterator[0]
        ret = self.get_document_by_id(curr_id)
        self.ids_iterator = self.ids_iterator[1:]
        return ret

    def _drop_duplicate_documents(self, documents: List[Document], index: Optional[str] = None) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID
        :param documents: A list of Document objects.
        :param index: name of the index
        :return: A list of Document objects.
        """
        _hash_ids = set([])
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already exists in index '%s'",
                    document.id,
                    index or self.index,
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents

    def _handle_duplicate_documents(
        self,
        documents: List[Document],
        index: Optional[str] = None,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.
        :param documents: A list of Document objects.
        :param index: name of the index
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :return: A list of Document objects.
        """

        index = index or self.index
        if duplicate_documents in ("skip", "fail"):
            documents = self._drop_duplicate_documents(documents, index)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=index, headers=headers)
            ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and duplicate_documents == "fail":
                raise DuplicateDocumentError(
                    f"Document with ids '{', '.join(ids_exist_in_db)} already exists in index = '{index}'."
                )

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents


class IEDocStore(IDocStore):
    """
    Elastic Document Store mask.
    """

    def __init__(
        self,
        client: Any,
        index: str = "document",
        label_index: str = "label",
        search_fields: Union[str, list] = "content",
        content_field: str = "content",
        name_field: str = "name",
        embedding_field: str = "embedding",
        embedding_dim: int = 768,
        custom_mapping: Optional[dict] = None,
        excluded_meta_data: Optional[list] = None,
        analyzer: str = "standard",
        recreate_index: bool = False,
        create_index: bool = True,
        refresh_type: str = "wait_for",
        similarity: str = "dot_product",
        return_embedding: bool = False,
        duplicate_documents: str = "overwrite",
        scroll: str = "1d",
        skip_missing_embeddings: bool = True,
        synonyms: Optional[List] = None,
        synonym_type: str = "synonym",
        batch_size: int = 10_000,
    ):
        super().__init__()

        self.client = client
        self._RequestError: Any = Exception

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        self.search_fields = search_fields
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.excluded_meta_data = excluded_meta_data
        self.analyzer = analyzer
        self.return_embedding = return_embedding

        self.custom_mapping = custom_mapping
        self.synonyms = synonyms
        self.synonym_type = synonym_type
        self.index: str = index
        self.label_index: str = label_index
        self.scroll = scroll
        self.skip_missing_embeddings: bool = skip_missing_embeddings
        self.duplicate_documents = duplicate_documents
        self.refresh_type = refresh_type
        self.batch_size = batch_size
        if similarity in ["cosine", "dot_product", "l2"]:
            self.similarity: str = similarity
        else:
            raise DocumentStoreError(f"Invalid value {similarity} for similarity, choose between 'cosine', 'l2' and 'dot_product'")
        client_info = self.client.info()
        self.server_version = tuple(int(num) for num in client_info["version"]["number"].split("."))

        self._init_indices(index=index, label_index=label_index, create_index=create_index, recreate_index=recreate_index)

    def _init_indices(self, index: str, label_index: str, create_index: bool, recreate_index: bool) -> None:
        if recreate_index:
            self._index_delete(index)
            self._index_delete(label_index)

        if not self._index_exists(index) and (create_index or recreate_index):
            self._create_document_index(index)

        if self.custom_mapping:
            logger.warning("Cannot validate index for custom mappings. Skipping index validation.")
        else:
            self._validate_and_adjust_document_index(index)

        if not self._index_exists(label_index) and (create_index or recreate_index):
            self._create_label_index(label_index)

    def _split_document_list(
        self, documents: Union[List[dict], List[Document]], number_of_lists: int
    ) -> Generator[Union[List[dict], List[Document]], None, None]:
        chunk_size = max((len(documents) + 1) // number_of_lists, 1)
        for i in range(0, len(documents), chunk_size):
            yield documents[i : i + chunk_size]

    @abc.abstractmethod
    def _do_bulk(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _do_scan(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        pass

    @abc.abstractmethod
    def _create_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        pass

    @abc.abstractmethod
    def _create_label_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        pass

    @abc.abstractmethod
    def _validate_and_adjust_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        pass

    @abc.abstractmethod
    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
        pass

    @abc.abstractmethod
    def _get_raw_similarity_score(self, score):
        pass

    def _bulk(
        self,
        documents: Union[List[dict], List[Document]],
        headers: Optional[Dict[str, str]] = None,
        refresh: str = "wait_for",
        _timeout: int = 1,
        _remaining_tries: int = 10,
    ) -> None:
        """
        Bulk index documents using a custom retry logic with
        exponential backoff and exponential batch size reduction to avoid overloading the cluster.

        The ingest node returns '429 Too Many Requests' when the write requests can't be
        processed because there are too many requests in the queue or the single request is too large and exceeds the
        memory of the nodes. Since the error code is the same for both of these cases we need to wait
        and reduce the batch size simultaneously.

        :param documents: List of documents to index
        :param headers: Optional headers to pass to the bulk request
        :param refresh: Refresh policy for the bulk request
        :param _timeout: Timeout for the exponential backoff
        :param _remaining_tries: Number of remaining retries
        """

        try:
            self._do_bulk(self.client, documents, refresh=self.refresh_type, headers=headers)
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:  # type: ignore
                logger.warning(
                    "Failed to insert a batch of '%s' documents because of a 'Too Many Requests' response. "
                    "Splitting the number of documents into two chunks with the same size and retrying in %s seconds.",
                    len(documents),
                    _timeout,
                )
                if len(documents) == 1:
                    logger.warning(
                        "Failed to index a single document. Your indexing queue on the cluster is probably full. Try resizing your cluster or reducing the number of parallel processes that are writing to the cluster."
                    )

                time.sleep(_timeout)

                _remaining_tries -= 1
                if _remaining_tries == 0:
                    raise DocumentStoreError("Last try of bulk indexing documents failed.")

                for split_docs in self._split_document_list(documents, 2):
                    self._bulk(
                        documents=split_docs,
                        headers=headers,
                        refresh=refresh,
                        _timeout=_timeout * 2,
                        _remaining_tries=_remaining_tries,
                    )
                return
            raise e

    # TODO: Add flexibility to define other non-meta and meta fields expected by the Document class
    def _create_document_field_map(self) -> Dict:
        return {self.content_field: "content", self.embedding_field: "embedding"}

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        index = index or self.index
        documents = self.get_documents_by_id([id], index=index, headers=headers)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Fetch documents by specifying a list of text id strings.

        :param ids: List of document IDs. Be aware that passing a large number of ids might lead to performance issues.
        :param index: search index where the documents are stored. If not supplied,
                      self.index will be used.
        :param batch_size: Maximum number of results for each query.
                           Limited to 10,000 documents by default.
                           To reduce the pressure on the cluster, you can lower this limit, at the expense
                           of longer retrieval times.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                        Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        index = index or self.index
        documents = []
        for i in range(0, len(ids), batch_size):
            ids_for_batch = ids[i : i + batch_size]
            query = {"size": len(ids_for_batch), "query": {"ids": {"values": ids_for_batch}}}
            if not self.return_embedding and self.embedding_field:
                query["_source"] = {"excludes": [self.embedding_field]}
            result = self._search(index=index, **query, headers=headers)["hits"]["hits"]
            documents.extend([self._convert_es_hit_to_document(hit) for hit in result])
        return documents

    def get_metadata_values_by_key(
        self,
        key: str,
        query: Optional[str] = None,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """
        Get values associated with a metadata key. The output is in the format:
            [{"value": "my-value-1", "count": 23}, {"value": "my-value-2", "count": 12}, ... ]

        :param key: the meta key name to get the values for.
        :param query: narrow down the scope to documents matching the query string.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param index: search index where the meta values should be searched. If not supplied,
                      self.index will be used.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        body: dict = {
            "size": 0,
            "aggs": {"metadata_agg": {"composite": {"sources": [{key: {"terms": {"field": key}}}]}}},
        }
        if query:
            body["query"] = {
                "bool": {"should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]}
            }
        if filters:
            if not body.get("query"):
                body["query"] = {"bool": {}}
            body["query"]["bool"].update({"filter": LogicalFilterClause.parse(filters).convert_to_elasticsearch()})
        result = self._search(**body, index=index, headers=headers)

        values = []
        current_buckets = result["aggregations"]["metadata_agg"]["buckets"]
        after_key = result["aggregations"]["metadata_agg"].get("after_key", False)
        for bucket in current_buckets:
            values.append({"value": bucket["key"][key], "count": bucket["doc_count"]})

        # Only 10 results get returned at a time, so apply pagination
        while after_key:
            body["aggs"]["metadata_agg"]["composite"]["after"] = after_key
            result = self._search(**body, index=index, headers=headers)
            current_buckets = result["aggregations"]["metadata_agg"]["buckets"]
            after_key = result["aggregations"]["metadata_agg"].get("after_key", False)
            for bucket in current_buckets:
                values.append({"value": bucket["key"][key], "count": bucket["doc_count"]})

        return values

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: Optional[int] = None,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Indexes documents for later queries.

        If a document with the same ID already exists:
        a) (Default) Manage duplication according to the `duplicate_documents` parameter.
        b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
        (This is only relevant if you pass your own ID when initializing a `Document`.
        If you don't set custom IDs for your Documents or just pass a list of dictionaries here,
        they automatically get UUIDs assigned. See the `Document` class for details.)

        :param documents: A list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
                          Optionally: Include meta data via {"content": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          You can use it for filtering and you can access it in the responses of the Finder.
                          Advanced: If you are using your own field mapping, change the key names in the dictionary
                          to what you have set for self.content_field and self.name_field.
        :param index: search index where the documents should be indexed. If you don't specify it, self.index is used.
        :param batch_size: Number of documents that are passed to the bulk function at each round.
                           If not specified, self.batch_size is used.
        :param duplicate_documents: Handle duplicate documents based on parameter options.
                                    Parameter options: ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicate documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: Raises an error if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to the client (for example {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                For more information, see [HTTP/REST clients and security](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html).
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """

        if index and not self._index_exists(index, headers=headers):
            self._create_document_index(index, headers=headers)

        if index is None:
            index = self.index

        batch_size = batch_size or self.batch_size

        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents, headers=headers
        )
        documents_to_index = []
        for doc in document_objects:
            index_message: Dict[str, Any] = {
                "_op_type": "index" if duplicate_documents == "overwrite" else "create",
                "_index": index,
                "_id": str(doc.id),
                # use _source explicitly to avoid conflicts with automatic field detection by ES/OS clients (e.g. "version")
                "_source": self._get_source(doc, field_map),
            }
            documents_to_index.append(index_message)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                self._bulk(documents_to_index, refresh=self.refresh_type, headers=headers)
                documents_to_index = []

        if documents_to_index:
            self._bulk(documents_to_index, refresh=self.refresh_type, headers=headers)

    def _get_source(self, doc: Document, field_map: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Document object to a dictionary that can be used as the "_source" field in an ES/OS index message."""

        _source: Dict[str, Any] = doc.to_dict(field_map=field_map)

        # cast embedding type as ES/OS cannot deal with np.array
        if isinstance(_source.get(self.embedding_field), np.ndarray):
            _source[self.embedding_field] = _source[self.embedding_field].tolist()

        # we already have the id in the index message
        _source.pop("id", None)

        # don't index query score and empty fields
        _source.pop("score", None)
        _source = {k: v for k, v in _source.items() if v is not None}

        # In order to have a flat structure in ES/OS + similar behavior to the other DocumentStores,
        # we "unnest" all value within "meta"
        _source.update(_source.pop("meta", None) or {})
        return _source

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 10_000,
    ):
        """Write annotation labels into document store.

        :param labels: A list of Python dictionaries or a list of Haystack Label objects.
        :param index: search index where the labels should be stored. If not supplied, self.label_index will be used.
        :param batch_size: Number of labels that are passed to the bulk function at each round.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        index = index or self.label_index
        if index and not self._index_exists(index, headers=headers):
            self._create_label_index(index, headers=headers)

        label_list: List[Label] = [Label.from_dict(label) if isinstance(label, dict) else label for label in labels]
        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(label_list, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(
                "Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                " This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                " the answer annotation and not the question."
                " Problematic ids: %s",
                ",".join(duplicate_ids),
            )
        labels_to_index = []
        for label in label_list:
            # create timestamps if not available yet
            if not label.created_at:
                label.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if not label.updated_at:
                label.updated_at = label.created_at

            index_message: Dict[str, Any] = {
                "_op_type": "index" if self.duplicate_documents == "overwrite" or label.id in duplicate_ids else "create",
                "_index": index,
            }

            _source = label.to_dict()

            # set id for elastic
            if _source.get("id") is not None:
                index_message["_id"] = str(_source.pop("id"))

            # use _source explicitly to avoid conflicts with automatic field detection by ES/OS clients (e.g. "version")
            index_message["_source"] = _source
            labels_to_index.append(index_message)

            # Pass batch_size number of labels to bulk
            if len(labels_to_index) % batch_size == 0:
                self._bulk(labels_to_index, refresh=self.refresh_type, headers=headers)
                labels_to_index = []

        if labels_to_index:
            self._bulk(labels_to_index, refresh=self.refresh_type, headers=headers)

    def update_document_meta(
        self, id: str, meta: Dict[str, str], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        if not index:
            index = self.index
        body = {"doc": meta}
        self._update(index=index, id=id, **body, refresh=self.refresh_type, headers=headers)

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index

        body: dict = {"query": {"bool": {}}}
        if only_documents_without_embedding:
            body["query"]["bool"]["must_not"] = [{"exists": {"field": self.embedding_field}}]

        if filters:
            body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        result = self._count(index=index, body=body, headers=headers)
        count = result["count"]
        return count

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        """
        Return the number of labels in the document store
        """
        index = index or self.label_index
        return self.get_document_count(index=index, headers=headers)

    def get_embedding_count(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the count of embeddings in the document store.
        """

        index = index or self.index

        body: dict = {"query": {"bool": {"must": [{"exists": {"field": self.embedding_field}}]}}}
        if filters:
            body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        result = self._count(index=index, body=body, headers=headers)
        count = result["count"]
        return count

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size, headers=headers
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        excludes = None
        if not return_embedding and self.embedding_field:
            excludes = [self.embedding_field]

        result = self._get_all_documents_in_index(
            index=index, filters=filters, batch_size=batch_size, headers=headers, excludes=excludes
        )
        for hit in result:
            document = self._convert_es_hit_to_document(hit)
            yield document


__all__ = ["IDocStore"]
