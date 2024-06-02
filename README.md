# PostLLM
### `llama.cpp` bindings for PostgreSQL

## Install
```sh
git clone --recurse-submodules https://github.com/albaike/postllm
```

### Build and install llama.cpp
```sh
cd llama.cpp
RUN cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON .
make
sudo make install
```

### Build postllm
```sh
cd ..
cmake .
make
```

### Install to postgresql
```sh
sudo make install
echo "shared_preload_libraries = 'postllm'" | sudo tee -a /etc/postgresql/16/main/postgresql.conf
```

### Restart db
```sh
sudo systemctl restart postgresql
```

## Test

### Download model for testing
```sh
cd files/
curl -O 'https://huggingface.co/ggml-org/gemma-1.1-7b-it-Q2_K-GGUF/resolve/main/gemma-1.1-7b-it.Q2_K.gguf?download=true'
```

### Docker
```sh
docker build -t postllm-test -f Dockerfile.test . && docker run -e POSTGRES_PASSWORD=pw postllm-test; echo $?
```