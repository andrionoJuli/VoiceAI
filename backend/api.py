import os
import torch
import nltk
import scipy
import numpy as np
nltk.download('punkt')

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from chat_service import Chat
from transformers import AutoProcessor, BarkModel
from optimum.bettertransformer import BetterTransformer
from datetime import datetime

from pathlib import Path
from utils import count_words, FileInfo, audio_generator
from pydub import AudioSegment


app = FastAPI()

# Configure GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

# Load TTS
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

# Improve processing speed
model = BetterTransformer.transform(model, keep_original_model=False)
model.enable_cpu_offload()

# Select voice
voice_preset = "v2/en_speaker_6"

chat_instance = Chat(prompt="You are a close companion to the users. Your role is to chat with the "
                            "user in an amicable manner. You aim to create a comfortable, "
                            "non-judgmental environment where the user can feels at ease and talk to "
                            "you as a friend. You are to respond casually in a warm, caring, and empathetic "
                            "manner. Seamlessly incorporate vocal inflections like 'I see', 'gotcha!', 'right!', "
                            "'oh dear', 'I understand', 'that makes sense', 'I hear you', and 'you know?' to "
                            "convey empathy and understanding when appropriate. Avoid responses like 'Ahaha', "
                            "'Hahaha', 'smile', 'nod', their synonyms or others words that cannot be expressed "
                            "verbally. You can use non-speech sounds but you can only incorporate the following "
                            "non-speech sounds : '[laughter]', '[laughs]', '[chuckle]', '[crying]', '[sighs]', "
                            "'[music]', '[gasps]', '[clears throat]', '...' for hesitations, and CAPITALIZATION "
                            "for emphasis of a word. Be concise and direct in your responses, you are to answer "
                            "as short as possible; if possible answer in one sentence else you are to answer in "
                            "three sentences with each sentence being not more than 20 words. Avoid repeating "
                            "yourself or providing unnecessary information and make sure to only ask one "
                            "question at a time if any.")


@app.post("/generate_audio_response/")
async def chat_with_assistant_audio(input_text: str):
    output = chat_instance(input_text)
    word_count = count_words(output)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_path = f"output/{chat_instance.session_id}"
    os.makedirs(directory_path, exist_ok=True)
    audio_path = f"{directory_path}/{timestamp}.mp3"
    if word_count <= 20:
        inputs = processor(output, voice_preset=voice_preset, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_array = model.generate(**inputs, do_sample=True, fine_temperature=0.3,
                                         coarse_temperature=0.7).cpu().numpy().squeeze()
        audio_32 = audio_array.astype(np.float32)
    else:
        texts = output.replace("\n", " ").strip()
        text_segments = nltk.sent_tokenize(texts)
        # Process text to audio
        # Approach 2 combined audio: risk longer processing time but easier streaming
        audio_segments = []
        for segment in text_segments:
            inputs = processor(segment, return_tensors="pt", voice_preset=voice_preset).to(device)
            audio_outputs = model.generate(**inputs, do_sample=True, fine_temperature=0.3,
                                           coarse_temperature=0.7)
            audio_segments.append(audio_outputs.cpu().numpy().squeeze())
        # Combine all audio
        combined_audio = np.concatenate(audio_segments)
        audio_32 = combined_audio.astype(np.float32)
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write("bark.wav", rate=sample_rate, data=audio_32)
    audio_segment = AudioSegment.from_wav("bark.wav")
    audio_segment.export(audio_path, format="mp3")
    file_info = FileInfo(session_id=chat_instance.session_id, timestamp=timestamp)
    return file_info


@app.get("/stream_audio/{session_id}/{timestamp}")
async def stream_audio(session_id: str, timestamp: str):
    audio_file_path = f"output/{session_id}/{timestamp}.mp3"
    if not Path(audio_file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    return StreamingResponse(audio_generator(audio_file_path), media_type="audio/mp3")


@app.get("/memory/")
async def get_memory():
    return chat_instance.memory
