import os
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Scope drive.file: chi thao tac tren file/thu muc do chinh app tao ra -> an toan
SCOPES = ["https://www.googleapis.com/auth/drive"]


def build_service(service_account_file: str):
    """
    Tao Google Drive service tu file Service Account JSON.

    Tham so:
    - service_account_file: duong dan toi file JSON (do tab Cai dat upload len).

    Tra ve:
    - doi tuong service de goi cac API Drive.
    """
    if not service_account_file or not os.path.exists(service_account_file):
        raise FileNotFoundError(
            f"Khong tim thay file Service Account JSON: {service_account_file}"
        )
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return service


def ensure_job_folder(service, parent_folder_id: str, job_name: str) -> str:
    """
    Tao mot thu muc con (theo ten job + thoi gian) ben trong folder cha de gom
    cac file ket qua cua mot lan chay.

    Tham so:
    - service: Drive service.
    - parent_folder_id: ID thu muc cha (da share cho service account). Co the rong.
    - job_name: ten goi y (thuong la ten video).

    Tra ve:
    - ID cua thu muc con vua tao. Neu khong co parent_folder_id, tra ve "" (upload thang vao folder cha mac dinh cua service account neu co).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{job_name}_{timestamp}"

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    folder = (
        service.files()
        .create(body=metadata, fields="id", supportsAllDrives=True)
        .execute()
    )
    return folder.get("id")


def upload_file(service, local_path: str, folder_id: str = "") -> dict:
    """
    Upload mot file len Google Drive.

    Tham so:
    - service: Drive service.
    - local_path: duong dan file can upload.
    - folder_id: ID thu muc dich (co the rong).

    Tra ve:
    - dict gom {name, id, link} cua file da upload.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Khong tim thay file de upload: {local_path}")

    file_name = os.path.basename(local_path)
    metadata = {"name": file_name}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaFileUpload(local_path, resumable=True)
    uploaded = (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id, name, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )
    return {
        "name": uploaded.get("name"),
        "id": uploaded.get("id"),
        "link": uploaded.get("webViewLink", ""),
    }
