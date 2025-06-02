# GenAiVoice-Chatbot_OiHackathon


Project Name: **GenAi Voice-Chatbot**

Group Name: **Oi Hackathon**

Main Participant Name: **Vishu Dua**

Team Participant Names:

- Parth Patel,  
- Shivang Ayar,
-    Keyur Desai

# Oi-Chatbot 🤖

A comprehensive voice-enabled chatbot system that combines speech-to-text, text generation, and text-to-speech capabilities. The system allows for natural voice interactions with an AI assistant, featuring real-time processing, voice activity detection, and natural language understanding.

## 🌟 Features

- **Speech-to-Text**: Real-time speech recognition using Faster Whisper
- **Text Generation**: AI-powered text responses using Mistral
- **Text-to-Speech**: Natural voice synthesis using TTS (Jenny model)
- **Voice Activity Detection**: Smart detection of speech segments
- **Real-time Processing**: Low-latency voice interaction
- **API Integration**: RESTful API endpoints for all services
- **Transcript Management**: Automatic conversation logging
- **Speaker Recognition**: Voice embedding and speaker identification

## 🏗️ Project Structure

```
Oi-Chatbot/
├── src/
│   ├── Speech_to_text/    # Speech recognition components
│   │   ├── api.py         # FastAPI endpoints
│   │   ├── audio.py       # Audio processing
│   │   ├── models.py      # Data models
│   │   ├── speaker.py     # Speaker recognition
│   │   ├── stt.py         # Speech-to-text core
│   │   ├── transcript.py  # Transcript management
│   │   └── utils.py       # Utility functions
│   │
│   ├── Text_GEN/         # Text generation components
│   │   ├── data/         # Training and model data
│   │   ├── modules/      # Custom modules
│   │   ├── transcripts/  # Conversation history
│   │   └── text_gen.py   # Text generation core
│   │
│   └── Text_To_Speech/   # Voice synthesis components
│       └── tts.py        # Text-to-speech core
│
├── Mistral/              # Mistral model files
├── Transcripts/          # Conversation transcripts
├── Requirements.txt      # Main project dependencies
└── stt.py               # Main speech-to-text script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended for better performance)
- Virtual environment (recommended) - 3 different venvs for each component:
  - Main venv: For core dependencies
  - Text_GEN venv: For text generation
  - TTS venv: For text-to-speech

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Oi-Chatbot.git
cd Oi-Chatbot
```

2. Create and activate the main virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install main dependencies:
```bash
pip install -r Requirements.txt
```

4. Set up Text Generation environment:
```bash
cd src/Text_GEN
python -m venv venv2
# On Windows
.\venv2\Scripts\activate
# On Linux/Mac
source venv2/bin/activate
pip install -r Requirements2.txt
```

5. Set up Text-to-Speech environment:
```bash
cd src/Text_To_Speech
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
pip install TTS
```

### Running the Services

1. Start the Text-to-Speech service:
```bash
cd src/Text_To_Speech
python tts.py
# Service will run on http://localhost:6969
```

2. Start the Speech-to-Text service:
```bash
python stt.py
# Service will run on http://localhost:8980
```

3. Start the Text Generation service:
```bash
cd src/Text_GEN
python text_gen.py
```

## 🎯 API Endpoints

### Text-to-Speech Service (Port 6969)
- `POST /tts`: Convert text to speech
  - Query parameter: `text` - Text to convert
  - Returns: Audio stream (WAV format)
- `POST /chat`: Get bot response and convert to speech
  - Query parameter: `text` - User input
  - Returns: Audio stream of bot response

### Speech-to-Text Service (Port 8980)
- Real-time speech recognition with voice activity detection
- Speaker recognition and identification
- Automatic transcript generation

## 🛠️ Technical Details

### Key Components

1. **Speech-to-Text**
   - Uses Faster Whisper for accurate speech recognition
   - Implements voice activity detection
   - Real-time audio processing
   - Speaker recognition using Resemblyzer
   - Automatic transcript management

2. **Text Generation**
   - Powered by Mistral model
   - Context-aware responses
   - Natural language understanding
   - Conversation history management
   - Custom response formatting

3. **Text-to-Speech**
   - Uses TTS with Jenny model
   - GPU acceleration support
   - High-quality voice synthesis
   - Streaming response support
   - Error handling and retries

### Dependencies

#### Main Dependencies
- numpy: Numerical computations
- torch: Deep learning framework
- sounddevice: Audio processing
- fastapi: API framework
- uvicorn: ASGI server
- requests: HTTP client
- resemblyzer: Voice embeddings
- scipy: Scientific computing
- faster_whisper: Speech recognition
- scikit-learn: Machine learning utilities
- pydantic: Data validation

#### Text Generation Dependencies
- transformers: Hugging Face transformers
- torch: PyTorch for model inference
- numpy: Numerical operations
- pandas: Data handling

#### Text-to-Speech Dependencies
- TTS: Coqui TTS engine
- torch: PyTorch for model inference
- numpy: Audio processing

## 📝 Usage Examples

1. Basic Text-to-Speech:
```bash
curl -X POST 'http://localhost:6969/tts?text=Hello%20world!'
```

2. Chat with Voice Response:
```bash
curl -X POST 'http://localhost:6969/chat?text=How%20are%20you?'
```

## 🔧 Troubleshooting

1. **Audio Device Issues**
   - Ensure microphone is properly connected
   - Check system audio settings
   - Verify sounddevice installation

2. **Model Loading Errors**
   - Verify CUDA installation for GPU support
   - Check model file paths
   - Ensure sufficient disk space

3. **API Connection Issues**
   - Verify all services are running
   - Check port availability
   - Ensure correct URL configurations


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Mistral AI for the language model
- Coqui TTS for the text-to-speech engine
- Faster Whisper for speech recognition
- Hugging Face for transformer models
- Resemblyzer for speaker recognition 
