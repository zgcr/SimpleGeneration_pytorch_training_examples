#!/bin/bash

export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=64
export HF_ENDPOINT=https://hf-mirror.com

MAX_RETRIES=50
RETRY_WAIT_SECONDS=600
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "========================================"
    echo "  Attempt $COUNT of $MAX_RETRIES"
    echo "========================================"
    echo ""

    hf download --repo-type dataset jasperai/monet --local-dir /root/autodl-tmp/huggingface_datasets/monet --max-workers 32

    if [ $? -eq 0 ]; then
        echo ""
        echo "Download completed successfully!"
        exit 0
    fi

    echo "Download failed, retrying in $((RETRY_WAIT_SECONDS / 60)) minutes..."
    sleep $RETRY_WAIT_SECONDS
done

echo "Max retries reached, exiting."
exit 1