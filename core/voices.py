import torch as tor
import pygame as pg
from qwen_tts import Qwen3TTSModel as qwen3
from typing import Literal, Optional, Union
import numpy as np
from .terminal import cprint
import gc
import atexit

from torch.cuda import OutOfMemoryError, AcceleratorError

# NOTE : we'll adjust dynamically if needed
pg.mixer.init(frequency=24000, size=-16, channels=1)

# Ryan      | English   | Dynamic male voice with strong rhythmic drive.  
# Aiden     | English   | Sunny American male voice with a clear midrange.  
# Ono_Anna  | Japanese  | Playful Japanese female voice with a light, nimble timbre.
# Vivian    | Chinese   | Bright, slightly edgy young female voice.  
# Serena    | Chinese   | Warm, gentle young female voice.  
# Uncle_Fu  | Chinese   | Seasoned male voice with a low, mellow timbre.  
# Dylan     | Chinese   | (Beijing Dialect) Youthful Beijing male voice with a clear, natural timbre.  
# Eric      | Chinese   | (Sichuan Dialect)  Lively Chengdu male voice with a slightly husky brightness.  
# Sohee     | Korean    | Warm Korean female voice with rich emotion.
class TTS:

    _shared_model = None

    def __init__(
            self,
            voice_language:Literal["Chinese","English","Japanese","Korean"] = "English",
            voice_speaker:Literal[
                "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Aiden", "Ryan", "Ono_Anna", "Sohee"
            ] = "Ryan",
            voice_instruct:Optional[str] = None
        ):

        # Efficiently load the model once into memory
        if TTS._shared_model is None:
            TTS._shared_model = qwen3.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map="cuda:0",
                dtype=tor.bfloat16
        )
        self.model = TTS._shared_model
        
        self.vl = voice_language
        self.vs = voice_speaker
        self.vi = voice_instruct

        atexit.register(self.cleanup)

        pass

    def ensure_mixer(self, sr, channels):
        current = pg.mixer.get_init()

        if current is None:
            pg.mixer.init(frequency=sr, size=-16, channels=channels)
            return

        cur_freq, cur_format, cur_channels = current

        if cur_freq != sr or cur_channels != channels:
            pg.mixer.quit()
            pg.mixer.init(frequency=sr, size=-16, channels=channels)

    @staticmethod
    def play_audio(audio):
        audio_int16 = (audio * 32767).astype(np.int16)
        sound = pg.sndarray.make_sound(audio_int16)
        sound.play()

    # Cleanup --------------------------------------

    def cleanup(self, destroy_model: bool = False):
        if destroy_model:
            self.destroy_model()

        gc.collect()

        if tor.cuda.is_available():
            tor.cuda.empty_cache()
            tor.cuda.ipc_collect()

    def destroy_model(self):
        try:
            del self.model
        except:
            pass

    @staticmethod
    def hard_reset():
        import os
        os._exit(0)

    # Safe Call -----------------------------------

    def safe_call(
        self,
        *args,
        retry: bool = True,
        hard_reset_on_fail: bool = False,
        **kwargs
    ):
        try:
            return self.__call__(*args, **kwargs)

        except (OutOfMemoryError, RuntimeError, AcceleratorError) as e:

            if "out of memory" not in str(e).lower():
                raise  # not a CUDA issue

            cprint("[TTS] CUDA OOM detected. Cleaning up...", "red")

            # 🔥 Clean GPU state
            self.cleanup(destroy_model=True)

            # 🔥 Force full cache clear
            if tor.cuda.is_available():
                tor.cuda.empty_cache()
                tor.cuda.ipc_collect()

            gc.collect()

            # 🔁 Retry once
            if retry:
                cprint("[TTS] Reloading model and retrying...", "yellow")

                TTS._shared_model = None
                self.model = qwen3.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    device_map="cuda:0",
                    dtype=tor.float16  # safer fallback
                )

                return self.safe_call(*args, retry=False, **kwargs)

            # 💣 If still failing
            cprint("[TTS] Retry failed.", "red")

            if hard_reset_on_fail:
                cprint("[TTS] Performing hard reset...", "red")
                self.hard_reset()

            raise

    def __call__(
            self,
            text:Union[list[str]|str],
            language:Optional[list[str]|str] = None,
            speaker:Optional[list[str]|str] = None,
            instruct:Optional[list[str]|str] = None
        ):

        language = language or self.vl
        speaker = speaker or self.vs
        instruct = instruct or self.vi
        
        with tor.no_grad():
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )

        wavs = [w.copy() for w in wavs]
        for audio in wavs:

            # 🔑 Convert float → int16 for pygame
            audio_int16 = (audio * 32767).astype(np.int16)

            # 🔑 Handle mono vs stereo
            channels = 1 if audio_int16.ndim == 1 else audio_int16.shape[1]

            self.ensure_mixer(sr, channels)
            # # 🔑 Re-init mixer if needed (important!)
            # pg.mixer.quit()
            # pg.mixer.init(frequency=sr, size=-16, channels=channels)

            # Create sound and play
            sound = pg.sndarray.make_sound(audio_int16)
            sound.play()

            # Wait for playback to finish
            # pg.time.wait(int(len(audio_int16) / sr * 1000))
            while pg.mixer.get_busy():
                pg.time.delay(10)

            del wavs  # after playback
            self.cleanup()
        
        return