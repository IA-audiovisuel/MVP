# Youtube transcription and diarization pipeline

Proof of Concept using yt-dlp to download a YouTube audio and WhisperX for transcription and diarization
https://github.com/yt-dlp/yt-dlp
https://github.com/m-bain/whisperX

## Setup

1. Download FFMPEG and add to PATH
Download and extract from https://www.gyan.dev/ffmpeg/builds/
Copy path to environment: system variables > Advanced > Environment Variables… > User variables Path > Edit... > New > C:\ffmpeg\bin

2. Get a HuggingFace Token and accept agreement to use pyannote gated models
https://huggingface.co/settings/tokens
https://huggingface.co/pyannote/segmentation-3.0
https://huggingface.co/pyannote/speaker-diarization-3.1

### FOR GPU INSTALLATION

3. Install CUDA toolkit 12.8 on your OS
Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
Windows: https://developer.nvidia.com/cuda-12-8-1-download-archive

4. Install python libraries for CUDA toolkit 12.8 and compatible Torch 2.8
pip install --upgrade setuptools pip wheel
pip install nvidia-cuda-runtime-cu12
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128