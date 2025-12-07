#  import libraries for enabling interaction with Hugging Face
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

# Import OS Library
import os


# store repo id and repo type
# since we are registering data - we need data space
repo_id = "harishsohani/MLOP-Project-Tourism"     # Name of space in Hugging Face is 'MLOP-Project-Tourism'
repo_type = "dataset"                             # sicne we want to upload the data - specify repo type as 'dataset'

# Initialize API client with Hugging Face Token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
# if not present then create
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# upload files from data folder
# here we use api to upload data file(s)
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print ("Uploaded Data file to Hugging Face Space (Dataset) Successfully")
