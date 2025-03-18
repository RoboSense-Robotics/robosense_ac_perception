#!/bin/bash
MODEL_URL="https://cdn.robosense.cn/AC_wiki/deepmodel.zip"
SAVE_PATH="modules/config"
EXPECTED_MD5="110fbb7322786f116b60b6e490b363e5"

mkdir -p "$(dirname "$SAVE_PATH")"

download_with_retry() {
    local retries=3
    local timeout=15

    for ((i=1; i<=retries; i++)); do
        echo "try download $i/$retries..."
        if wget -P "$SAVE_PATH" "$MODEL_URL"; then
            echo "download success!"
            return 0
        else
            echo "download failed, retrying..."
            sleep $timeout
        fi
    done
    return 1
}

if download_with_retry; then
    ACTUAL_MD5=$(md5sum "$SAVE_PATH"/deepmodel.zip | awk '{print $1}')

    if [ -n "$EXPECTED_MD5" ]; then
        if [ "$ACTUAL_MD5" == "$EXPECTED_MD5" ]; then
            echo "MD5 verified success：$ACTUAL_MD5"
            unzip -o "$SAVE_PATH"/deepmodel.zip -d "$SAVE_PATH"
        else
            echo "MD5 verified failed!"
            echo "expected: $EXPECTED_MD5"
            echo "actual: $ACTUAL_MD5"
            exit 1
        fi
    fi
else
    echo "download failed!"
    exit 1
fi
