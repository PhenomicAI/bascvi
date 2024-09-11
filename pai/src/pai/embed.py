import json
import logging
import os
import time
import zipfile
import hashlib
import math
import requests

BACKEND_API_URI = "https://backend-api.scref.phenomic.ai"
CHUNK_SIZE = 2**20  # 1 Megabyte

# https://docs.hdfgroup.org/hdf5/v1_14/_f_m_t3.html#Superblock
H5AD_SIGNATURE = bytes.fromhex("894844460d0a1a0a")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)


class PaiEmbeddings:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def download_example_h5ad(self):
        logger.info("Downloading example h5ad")
        url = BACKEND_API_URI + "/download_example_h5ad"

        response = requests.get(url)

        adatas_dir = os.path.join(self.tmp_dir, "adatas")
        if not os.path.exists(adatas_dir):
            os.mkdir(adatas_dir)

        file_path = os.path.join(adatas_dir, "anndata_example.h5ad")
        with open(file_path, "wb") as binary_file:
            binary_file.write(response.content)

    def inference(self, h5ad_path, tissue_organ):
        assert h5ad_path.endswith(".h5ad")
        assert os.path.exists(h5ad_path)

        job_id = self.upload_h5ad(h5ad_path, tissue_organ)
        self.listen_job_status(job_id)
        self.download_job(job_id)

    def get_upload_uuid(self, chunks):
        logger.info("Getting upload id")
        url = BACKEND_API_URI + "/start_upload"
        response = requests.post(url, json={"chunk_count": chunks})

        if response.ok:
            self.upload_uuid = json.loads(response.json())["uuid"]
            logger.info(f"Recieved uuid: {self.upload_uuid}")
        else:
            raise Exception("Upload uuid not recieved", response)

    def upload_chunks(self, chunks, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as file:
            for i in chunks:
                chunk = file.read(CHUNK_SIZE)
                hash_md5.update(chunk)
                response = requests.post(
                    BACKEND_API_URI + "/upload_chunk",
                    data={"chunk_id": i, "uuid": self.upload_uuid},
                    files={"file": chunk},
                )
        return hash_md5.hexdigest()

    def upload_h5ad(self, h5ad_path, tissue_organ):
        logger.info("Checking destination folders...")

        zips_dir = os.path.join(self.tmp_dir, "zips")
        if not os.path.exists(zips_dir):
            os.mkdir(zips_dir)

        results_dir = os.path.join(self.tmp_dir, "results")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        logger.info("Uploading h5ad file...")
        size = os.path.getsize(h5ad_path)
        chunks = math.ceil(size / CHUNK_SIZE)
        self.get_upload_uuid(chunks)

        check_h5ad_signature(h5ad_path)
        hash = self.upload_chunks(range(chunks), h5ad_path)
        job_data = {
            "uuid": self.upload_uuid,
            "hash": hash,
            "tissueOrgan": tissue_organ,
        }
        response = requests.post(
            BACKEND_API_URI + "/upload_status",
            json=job_data,
        )

        if response.status_code == 200:
            job_id = json.loads(response.content)["id"]
            logger.info(f"Upload complete, job id: {job_id}")
            return job_id
        elif response.status_code == 201:
            # TODO Handle missing chunks
            pass
        else:
            raise Exception(response.status_code, response.reason)

    def get_job_status(self, job_id):
        url = BACKEND_API_URI + "/job"  # TODO
        params = {"job_id": job_id}

        response = requests.get(url, params=params)

        if response.status_code >= 200 and response.status_code < 300:
            status = json.loads(response.content)["status"]
            logger.info(f"Job status: {status}")
            return status
        else:
            raise Exception(response.status_code, response.reason)

    def listen_job_status(self, job_id):
        logger.info("Listening for job status")
        while True:
            status = self.get_job_status(job_id)
            if status in ["SUBMITTED", "VALIDATING", "RUNNING"]:
                time.sleep(5)  # sleep 5s
                continue
            elif status in ["COMPLETED", "FAILED", "ERROR"]:
                break

    def download_job(self, job_id):
        logger.info("Downloading job")
        url = BACKEND_API_URI + "/download"
        data = {"job_id": job_id}

        response = requests.post(url, json=data)

        zips_dir = os.path.join(self.tmp_dir, "zips")
        results_dir = os.path.join(self.tmp_dir, "results")

        zip_path = os.path.join(zips_dir, f"{job_id}.zip")
        job_dir = os.path.join(results_dir, job_id)

        with open(zip_path, "wb") as binary_file:
            binary_file.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(job_dir)


def check_h5ad_signature(file_path):
    with open(file_path, "rb") as file:
        signature = file.read(8)
        if signature != H5AD_SIGNATURE:
            logger.error("H5AD Signature mismatch")
            raise Exception("H5AD file does not match signature")

        # TODO consider option to cleanup zip file
