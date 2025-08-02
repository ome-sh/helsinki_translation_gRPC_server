# Spanish Translation Service

A high-performance gRPC translation service for Spanish to English and Chinese with intelligent dynamic batching and GPU acceleration.

## Features

- **Multi-Target Translation**: Spanish → English & Spanish → Chinese
- **Dynamic Batching**: 50ms intelligent batching for optimal throughput
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Language-Specific Batching**: Separate batching queues for each target language
- **Production Ready**: Robust error handling, timeout management, and logging
- **MarianMT Models**: High-quality neural machine translation

## Quick Start

### Installation

```bash
pip install grpcio grpcio-tools torch transformers
```

### Generate gRPC Files

```bash
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. translation.proto
```

### Run the Service

```bash
python translation_server.py
```

The service will start on `[::]:50052`

## API Endpoints

### Spanish to English Translation

```python
# Request: TranslationRequest
{
  "text": "Hola, ¿cómo estás?"
}

# Response: TranslationResponse
{
  "translated_text": "Hello, how are you?"
}
```

### Spanish to Chinese Translation

```python
# Request: TranslationRequest
{
  "text": "Hola, ¿cómo estás?"
}

# Response: TranslationResponse
{
  "translated_text": "你好，你好吗？"
}
```

## Performance Features

- **Dynamic Batching**: Max 30 requests per batch, 50ms timeout
- **Language Separation**: English and Chinese requests batched separately
- **GPU Optimization**: Automatic device detection and tensor placement
- **Beam Search**: 4-beam search with length penalty for quality
- **Concurrent Processing**: Multi-threaded batch processing

## Configuration

- **Models**: 
  - ES→EN: `Helsinki-NLP/opus-mt-es-en`
  - ES→ZH: `Helsinki-NLP/opus-tatoeba-es-zh`
- **Max Batch Size**: 30 requests (configurable)
- **Batch Timeout**: 50ms (configurable)
- **Max Length**: 512 tokens
- **Port**: 50052 (default)

## Example Usage

```python
import grpc
import translation_pb2
import translation_pb2_grpc

# Connect to service
channel = grpc.insecure_channel('localhost:50052')
stub = translation_pb2_grpc.TranslationServiceStub(channel)

# Translate to English
request = translation_pb2.TranslationRequest(text="¡Hola mundo!")
response = stub.TranslateToEnglish(request)
print(f"English: {response.translated_text}")

# Translate to Chinese
response = stub.TranslateToChinese(request)
print(f"Chinese: {response.translated_text}")
```

## Architecture

- **Dynamic Batcher**: Collects requests and processes them in optimized batches
- **Language-Specific Processing**: Separate pipelines for English and Chinese
- **Thread Pool**: Concurrent batch processing with 10 worker threads
- **Device Management**: Automatic GPU/CPU detection and tensor placement

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- gRPC
- CUDA (optional, for GPU acceleration)

## License

AGPL-3.0 (MarianMT models retain their original licenses)
