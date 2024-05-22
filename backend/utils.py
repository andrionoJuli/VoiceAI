import tempfile

from pydantic import BaseModel
from pydub import AudioSegment


def count_words(text):
    words = text.split()
    return len(words)


def audio_generator(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        chunk = audio_file.read(1024)
        while chunk:
            yield chunk
            chunk = audio_file.read(1024)


def convert_wav_to_mp3(wav_filename):
    mp3_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_segment = AudioSegment.from_wav(wav_filename)
    audio_segment.export(mp3_temp_file.name, format="mp3")
    return mp3_temp_file


class FileInfo(BaseModel):
    session_id: str
    timestamp: str

# Approach 1 batching: Great processing speed but complicate get request
        # inputs = processor(text_segments, return_tensors="pt", voice_preset=voice_preset).to(device)
        # audio_outputs = model.generate(**inputs, do_sample=True, fine_temperature=0.3,
        #                                coarse_temperature=0.7)
        # sample_rate = model.generation_config.sample_rate
        # audio_paths = []
        # for i, audio in enumerate(audio_outputs):
        #     output_filename = directory_path + f"/{timestamp}_{i}.wav"
        #     audio_32 = audio.to(torch.float32)
        #     scipy.io.wavfile.write(output_filename, rate=sample_rate, data=audio_32.cpu().numpy().squeeze())
        #     audio_paths.append(output_filename)
        # file_responses = []
        # for audio_path in audio_paths:
        #     file_responses.append(FileResponse(audio_path, media_type="audio/wav"))
        # return file_responses
