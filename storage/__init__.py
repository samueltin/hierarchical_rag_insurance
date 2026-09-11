"""
Azure Blob Storage access, shared by the ingestion pipeline and the chat service.

Kept deliberately free of pipeline and retrieval dependencies: it needs only
azure-storage-blob and azure-identity, so either side can import it without
pulling in the other's stack.
"""

from .blob_storage import (
    JSON_CONTENT_TYPE,
    MARKDOWN_CONTENT_TYPE,
    BlobLocation,
    account_url,
    blob_client,
    blob_exists,
    container_client,
    delete_blobs,
    download_blob,
    download_json,
    ensure_container,
    list_blobs,
    parse_blob_url,
    service_client,
    upload_bytes,
    upload_json,
    upload_text,
)

__all__ = [
    "JSON_CONTENT_TYPE",
    "MARKDOWN_CONTENT_TYPE",
    "BlobLocation",
    "account_url",
    "blob_client",
    "blob_exists",
    "container_client",
    "delete_blobs",
    "download_blob",
    "download_json",
    "ensure_container",
    "list_blobs",
    "parse_blob_url",
    "service_client",
    "upload_bytes",
    "upload_json",
    "upload_text",
]
