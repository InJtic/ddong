from huggingface_hub import HfApi, create_repo


def upload_to_hf(data_dir: str, repo_id: str, token: str | None = None):
    api = HfApi(token=token)

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"Uploading files from {data_dir}/*")

    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=".",
        token=token,
    )


if __name__ == "__main__":
    upload_to_hf(
        data_dir="./data",
        repo_id="Jtic/ddong-data",
    )
