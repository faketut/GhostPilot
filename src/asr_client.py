import asyncio
import logging
import azure.cognitiveservices.speech as speechsdk
from src.config import config

logger = logging.getLogger(__name__)

class ASRClient:
    def __init__(self, subscription_key: str, region: str = "", endpoint: str = ""):
        self.subscription_key = subscription_key
        self.region = region
        self.endpoint = endpoint
        self._is_running = False
        self.transcriber = None
        self.push_stream = None

    async def start_streaming(self, audio_queue: asyncio.Queue, text_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """
        Reads from audio_queue and writes to Azure PushAudioInputStream.
        Emits partial and final transcripts to text_queue.
        """
        if not self.subscription_key or not self.region or "your-azure" in self.subscription_key:
            logger.warning("Azure Speech Key/Region not configured. ASR will not run.")
            return

        self._is_running = True
        logger.info("Azure ASR streaming starting...")

        # Setup Azure Speech Config
        if self.endpoint:
            speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, endpoint=self.endpoint)
        else:
            speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        speech_config.speech_recognition_language = config.ASR_LANGUAGE
        
        # Enable diarization for intermediate results
        speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, value='true')

        # Setup Audio Stream
        # 16000 Hz, 16-bit, mono
        audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
        self.push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        # Setup Transcriber
        self.transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)

        # Callbacks
        def transcribing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            text = evt.result.text
            # 优化手段一：流式增量输出，利用 length/confidence 阈值防止下游频繁抖动
            if text and len(text) > 8:
                speaker_id = evt.result.speaker_id if evt.result.speaker_id else "Unknown"
                loop.call_soon_threadsafe(text_queue.put_nowait, {
                    "type": "partial", 
                    "text": text,
                    "speaker": speaker_id
                })

        def transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                speaker_id = evt.result.speaker_id if evt.result.speaker_id else "Unknown"
                loop.call_soon_threadsafe(text_queue.put_nowait, {
                    "type": "final", 
                    "text": evt.result.text,
                    "speaker": speaker_id
                })
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                pass # Silence or noise

        def session_stopped_cb(evt: speechsdk.SessionEventArgs):
            logger.info("Azure ASR session stopped.")

        def canceled_cb(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
            logger.error(f"Azure ASR Canceled: {evt.reason}. Error details: {evt.error_details}")

        # Connect callbacks
        self.transcriber.transcribing.connect(transcribing_cb)
        self.transcriber.transcribed.connect(transcribed_cb)
        self.transcriber.session_stopped.connect(session_stopped_cb)
        self.transcriber.canceled.connect(canceled_cb)

        # Start continuous recognition
        self.transcriber.start_transcribing_async()

        # Audio feeding loop
        while self._is_running:
            try:
                pcm_chunk = await audio_queue.get()
                if self.push_stream:
                    # Write PCM chunk to Azure
                    self.push_stream.write(pcm_chunk)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ASR Feed Error: {e}")

        # Cleanup
        if self.push_stream:
            self.push_stream.close()
        if self.transcriber:
            self.transcriber.stop_transcribing_async()
        logger.info("Azure ASR streaming stopped.")

    def stop(self):
        self._is_running = False
