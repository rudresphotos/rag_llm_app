import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
container_name = os.environ["AZURE_STORAGE_CONTAINER"]
logs_container_name = os.environ["AZURE_LOGS_CONTAINER"]

# --- Blob Clients ---
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# --- Upload a File ---
def upload_file_to_blob(local_file_path, blob_name):
    with open(local_file_path, "rb") as f:
        container_client.upload_blob(name=blob_name, data=f, overwrite=True)

# --- Download Single Blob ---
def download_blob_to_file(blob_name, local_file_path):
    with open(local_file_path, "wb") as f:
        download_stream = container_client.download_blob(blob_name)
        f.write(download_stream.readall())

# --- List All Blobs in Container ---
def list_blobs_in_container():
    return [blob.name for blob in container_client.list_blobs()]

# --- Download All Docs to Local (Cleans old files) ---
def download_all_docs_to_local(local_folder="docs"):
    # Remove old files
    if os.path.exists(local_folder):
        for file in os.listdir(local_folder):
            file_path = os.path.join(local_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(local_folder, exist_ok=True)

    # Download fresh files
    for blob_name in list_blobs_in_container():
        if blob_name.endswith((".pdf", ".txt", ".docx", ".md")):
            local_path = os.path.join(local_folder, blob_name)
            download_blob_to_file(blob_name, local_path)

# --- Upload Chat Log to Logs Container ---
def upload_conversation_log(log_data, filename):
    logs_container = blob_service_client.get_container_client(logs_container_name)
    logs_container.upload_blob(name=filename, data=log_data, overwrite=True)
