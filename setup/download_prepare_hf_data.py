# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    
    # Add token handling
    token = os.getenv("HF_TOKEN")
    if token is None:
        print("Warning: HF_TOKEN not found in environment. Some datasets might be inaccessible.")
        print("Please run: export HF_TOKEN=your_huggingface_token")
    
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                token=token,
                resume_download=True,
                max_workers=4,
                tqdm_class=None,
                revision="main",
            )
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                print(f"Error occurred: {e}")
                print(f"Attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to download after {max_retries} attempts")
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(dataset, memory, data_dir, seed=42, nchunks=32):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
        "pile": "EleutherAI/pile",
        "pile_deduplicated": "EleutherAI/the_pile_deduplicated",
        "pile_uc": "monology/pile-uncopyrighted",
        "c4": "allenai/c4",
        "redpajama": "togethercomputer/RedPajama-Data-1T",
        "refineweb": "tiiuae/falcon-refinedweb",
        "slimpajama": "togethercomputer/SlimPajama-627B"
    }[dataset]

    # Use temporary directory for initial processing
    tmp_src_dir = f"/tmp/{dataset}"
    tmp_out_dir = f"/tmp/{dataset}_shuffled"
    
    # Final destination directories
    final_src_dir = f"{data_dir}/{dataset}"
    final_out_dir = f"{data_dir}/{dataset}_shuffled"
    
    os.makedirs(tmp_src_dir, exist_ok=True)
    os.makedirs(tmp_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_src_dir), exist_ok=True)

    # Rest of the configurations remain exactly the same
    work_dir = tmp_src_dir
    prefix = f"{dataset}.chunk."

    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
        "pile": ".jsonl.zst",
        "pile_deduplicated": ".jsonl.zst",
        "pile_uc": ".jsonl.zst",  # Add this line
        "c4": ".json.gz",
        "redpajama": ".jsonl",
        "refineweb": ".jsonl",
        "slimpajama": ".jsonl"
    }[dataset]

    cat_command = {
        "fineweb_edu": "cat",
        "fineweb_edu_10bt": "cat",
        "dclm_baseline_1.0": "zstdcat",
        "dclm_baseline_1.0_10prct": "zstdcat",
        "pile": "zstdcat",  
        "pile_deduplicated": "zstdcat",
        "pile_uc": "zstdcat",
        "c4": "gunzip -c",
        "redpajama": "cat",
        "refineweb": "cat",
        "slimpajama": "cat"
    }[dataset]

    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
        "pile": "*.jsonl.zst",
        "pile_deduplicated": "*.jsonl.zst",
        "pile_uc": "*.jsonl.zst",
        "c4": "en/*.json.gz",
        "redpajama": "*.jsonl",
        "refineweb": "*.jsonl",
        "slimpajama": "*.jsonl"
    }[dataset]

    suffix = ".jsonl"
    k_validation = 10000  # Number of lines to take from each chunk for validation

    try:
        # Download and process in tmp directory
        terashuf_dir = setup_terashuf(work_dir)
        download_dataset(repo_id, tmp_src_dir, allow_patterns)

        if "fineweb" in dataset:
            parquet_to_jsonl(dataset, work_dir, tmp_src_dir, tmp_src_dir)

        # Rest of the processing remains the same but uses tmp directories
        os.environ["MEMORY"] = f"{memory}"
        os.environ["SEED"] = f"{seed}"

        terashuf_executable = os.path.join(terashuf_dir, "terashuf")
        run_command(
            f"ulimit -n 100000 && "
            f"find {tmp_src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 {cat_command} | {terashuf_executable} | "
            f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {tmp_out_dir}/{prefix}"
        )

        # Move final results to destination
        print(f"Moving processed data to {final_out_dir}")
        if os.path.exists(final_out_dir):
            run_command(f"rm -rf {final_out_dir}")
        run_command(f"mv {tmp_out_dir} {final_out_dir}")

    finally:
        # Cleanup
        print("Cleaning up temporary directories...")
        # run_command(f"rm -rf {tmp_src_dir}")
        # if os.path.exists(tmp_out_dir):
        #     run_command(f"rm -rf {tmp_out_dir}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed, args.nchunks)
