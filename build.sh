#!/usr/bin/env bash
# Build rknn_vad helper for RV1106.
#
# On Linux x86_64: runs natively (requires cmake and make).
# On macOS or non-x86_64 hosts: runs inside a Linux x86_64 container so the
# arm-rockchip830 toolchain (Linux x86_64 ELF) can execute.
set -e

cd "$(dirname "$0")"

if [ ! -d third_party/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin ]; then
    echo "Toolchain missing. Run: git submodule update --init --recursive"
    exit 1
fi

build_native() {
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-arm-rockchip830.cmake
    cmake --build build -j
}

build_in_docker() {
    # Run apt-get as root, then drop privileges to host user for cmake.
    # `chown -R` at the end ensures build/ is owned by the host user.
    UID_HOST=$(id -u)
    GID_HOST=$(id -g)
    docker run --platform linux/amd64 --rm \
        -v "$(pwd):/work" -w /work \
        -e UID_HOST="$UID_HOST" -e GID_HOST="$GID_HOST" \
        debian:bookworm-slim \
        bash -c '
            set -e
            apt-get update >/dev/null
            apt-get install -y --no-install-recommends cmake make >/dev/null
            cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-arm-rockchip830.cmake
            cmake --build build -j
            chown -R "$UID_HOST:$GID_HOST" build
        '
}

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)
        build_native
        ;;
    *)
        echo "Building inside Linux x86_64 container..."
        build_in_docker
        ;;
esac

echo ""
echo "Built: build/bin/rknn_vad"
file build/bin/rknn_vad
ls -lh build/bin/rknn_vad
