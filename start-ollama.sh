#!/bin/sh

MODEL_DIR='root/.ollama/models/manifests/registry.ollama.ai/library/llama3'

ollama serve &

echo 'Waiting for Ollama service to start...'
sleep 30

if [ ! "$(ls -A $MODEL_DIR)" ]; then
  echo 'Llama3 model not found, downloading...'
  ollama pull llama3
  echo 'Model downloaded successfully.'
else
  echo 'Llama3 model already present, skipping download.'
fi

tail -f /dev/null