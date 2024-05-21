import torch
import nltk

from streamlit_mic_recorder import speech_to_text
from transformers import AutoProcessor, BarkModel
from optimum.bettertransformer import BetterTransformer
from backend.chat_service import Chat
from streamlit_float import *

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

state = st.session_state


def count_words(text):
    words = text.split()
    return len(words)


# initializing session state & prompts
def set_main():
    if "chat" not in state:
        state.chat = Chat(prompt="You are a close companion to the users. Your role is to chat with the "
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
    if "memory" not in state:
        state.memory = []

    if 'my_stt_output' not in state:
        state.my_stt_output = []


# incorporate callback for Speech to Text
def callback():
    if state.my_stt_output:
        st.write(state.my_stt_output)


def main():
    set_main()

    # Show the chat
    for message in st.session_state.memory:
        if message["role"] != "system":
            with st.chat_message(message["role"]):  # The code that shows the chat
                st.markdown(message["content"])

    if text:
        # Add the user text to the history
        state.memory.append({"role": "user", "content": text})
        # Show the processed text text
        with st.chat_message("user"):
            st.write(text)

        # Predict and output the response
        with st.chat_message("assistant"):
            output = state.chat(text)
            state.memory.append({"role": "assistant", "content": output})
            with st.spinner("Generate response..."):
                st.write(output)
            word_count = count_words(output)
            if word_count <= 20:
                inputs = processor(output, voice_preset=voice_preset, return_tensors="pt").to(device)
                with torch.no_grad():
                    audio_array = model.generate(**inputs, do_sample=True, fine_temperature=0.3, coarse_temperature=0.7).cpu().numpy().squeeze()
                sample_rate = model.generation_config.sample_rate
                # Display the audio
                st.audio(audio_array, sample_rate=sample_rate)
            else:
                texts = output.replace("\n", " ").strip()
                text_segments = nltk.sent_tokenize(texts)
                # Process text to audio
                inputs = processor(text_segments, return_tensors="pt", voice_preset=voice_preset).to(device)
                audio_outputs = model.generate(**inputs, do_sample=True, fine_temperature=0.3, coarse_temperature=0.7).cpu().numpy().squeeze()
                # for segment in text_segments:
                #     inputs = processor(segment, return_tensors="pt", voice_preset=voice_preset).to(device)
                #     audio_outputs = model.generate(**inputs, do_sample=True, fine_temperature=0.3,
                #                                    coarse_temperature=0.7)
                #     audio_segments.append(audio_outputs.squeeze().cpu().numpy())

                # Combine all audio
                #combined_audio = np.concatenate(audio_segments)
                sampling_rate = model.generation_config.sample_rate
                for audio in audio_outputs:
                    st.audio(audio, sample_rate=sampling_rate)
            pass


if __name__ == '__main__':
    st.title("Llama 3 8B Voice AI")
    # Create a container to store the STT button
    footer_container = st.container()
    with footer_container:
        # The button that record the audio and convert it to text
        text = speech_to_text(language='en', use_container_width=True, callback=callback, key='STT')
    # Set the record audio button at the bottom
    footer_container.float("bottom: 0rem;")
    main()
