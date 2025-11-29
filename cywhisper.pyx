# cython: embedsignature=True  
# cython: embedsignature.format=python

import time
from typing import Coroutine
import os
import logging
from libcpp.vector cimport vector
from libc.stdint cimport int_fast64_t
from datetime import datetime
import asyncio
import concurrent.futures

cimport cywhisper

# setting logger
logger = logging.getLogger(__name__)

# Disable message sending to root logger
logger.propagate = False


cdef class Whisper:  
    cdef audio_async *_audio  
    cdef whisper_context *_ctx  
    cdef whisper_full_params _full_params

    cdef bint _is_running
    cdef bint _is_break  
    cdef bint _is_completed

    cdef bint _no_context
    cdef int _length_ms
    cdef int _keep_ms
    cdef int _step_ms
    cdef string _whisper_path
    cdef string _vad_path

    cdef bint use_vad  
    cdef bint save_audio
    cdef bint no_timestamps
    cdef string filename_out
    cdef int_fast64_t stop_time
    cdef int_fast64_t start_time
    cdef bint no_fallback

    cdef object _recognize_corout
    cdef object _recognize_future
    cdef object _loop

    @property
    def is_awaitable(self) -> bool:
        """Show whether async recognize is available"""

        return  not self._is_completed \
                and self._recognize_future is not None \
                and isinstance(self._recognize_future, concurrent.futures.Future) \
                and not self._recognize_future.cancelled()
      
    def __init__(self, int length_ms, str language, str whisper_model_path, str vad_model_path, bool save_audio=False, 
                    str filename_out="", bool print_timestamps = True, int logger_level = 40, 
                    int step_ms=0, async_loop: asyncio.AbstractEventLoop = None):
        """To load whisper model, default async functions is available. If async_loop is not None, asyncio run to set async_loop, otherwise
            that's run to event loop where class's example was created. 
        Inputs:
            length_ms: int
                max length of audio buffer that keeps micro input
            language: str
                recognizing language
            whisper_model_path: str
                absolute path to whisper model include name
            vad_model_path: str
                absolute path to vad model include name
            save_audio: bool
                If it's True then will be save recording audio to .wav file
            filename_out: str
                Name of file to write recognized text. Default is empty, so doesn't write
            print_timestamps: bool
                Before every chunk will be timestamps that show begin and end of chunk within recorded audio data
            logger_level: int
                Level for logging. Default is logging.ERROR
            step_ms: int
                processing interval in milliseconds. Determines how often the recognition loop process audio from the buffer. Default as the same length_ms
            async_loop: asyncio.AbstractEventLoop
                event loop to run async functions
        """

        # logger settings
        logger.setLevel(logger_level)
        file_handler = logging.FileHandler("cywhisper.log")
        file_handler.setLevel(logger_level)
        formatter = logging.Formatter('%(module)8s | %(asctime)s | %(filename)11s | %(funcName)20s:%(lineno)3d |  %(levelname)6s | %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if async_loop is None:
            self._loop = asyncio.get_event_loop()
        else:
            self._loop=async_loop

        # Convert python string to c++ string
        cdef bytes cpp_language = language.encode('utf-8')  
          
        self._is_break = False
        self._is_running = False
        self._is_completed = True

        self.use_vad = True   
        self.save_audio = save_audio
        self.filename_out = filename_out.encode('utf-8')

        self._length_ms = length_ms
        self.no_timestamps = not print_timestamps
        self._keep_ms = 200
        self._no_context = False

        # time count for audio get in recognize loop
        self._step_ms = self._length_ms if step_ms == 0 else step_ms

        self._whisper_path = whisper_model_path.encode('utf-8')
        self._vad_path = vad_model_path.encode('utf-8')
          
        cdef int capture_id = -1 
        self._audio = new audio_async(self._length_ms)  
        if not self._audio.init(capture_id, WHISPER_SAMPLE_RATE):
            logger.error("Audio init is failed!")
            raise ValueError("Audio init is failed!")

        if (string(cpp_language) != string(b"auto") and whisper_lang_id(cpp_language) == -1):
            logger.error("Undefined language")
            raise ValueError("Undefined language")

        ggml_backend_load_all()  
           
        cdef whisper_context_params cparams = whisper_context_default_params()
        cparams.use_gpu = True
        cparams.flash_attn = True
        
        # cdef bytes whisper_path = self.whisper_path.encode('utf-8')
        self._ctx = whisper_init_from_file_with_params(  
            self._whisper_path.c_str(),   
            cparams  
        )  
        if self._ctx == NULL:  
            raise RuntimeError("Failed to initialize whisper context") 

        cdef whisper_vad_params vad_params = whisper_vad_default_params()
        vad_params.threshold = 0.5
        vad_params.min_speech_duration_ms = 250
        vad_params.min_silence_duration_ms = 100
        vad_params.max_speech_duration_s = float('inf')
        vad_params.speech_pad_ms = 30
        vad_params.samples_overlap = 0.1
          
        self._full_params = whisper_full_default_params(whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY) 
        self._full_params.language = cpp_language 
        self._full_params.n_threads = min(4, os.cpu_count() or 4)
        self._full_params.print_progress = True if logger_level == logging.DEBUG else False
        self._full_params.print_realtime = False
        self._full_params.print_special = False
        self._full_params.print_timestamps = True if logger_level == logging.DEBUG else False
        self._full_params.translate = False
        self._full_params.single_segment = not self.use_vad
        self._full_params.max_tokens = 32
        self._full_params.beam_search.beam_size = -1
        self._full_params.audio_ctx = 0
        self._full_params.tdrz_enable = False

        self._full_params.temperature_inc = 0.0 if self.no_fallback else self._full_params.temperature_inc
          
        if self.use_vad:  
            self._full_params.vad = True  
            self._full_params.vad_model_path = self._vad_path.c_str()
            self._full_params.vad_params = vad_params
      
    def __dealloc__(self): 
        self.close()
      
    def stop_recognize(self):  
        if self._is_running:  
            self._is_running = False
            self.stop_time = <int_fast64_t>(round(time.time() * 1000, 3))
            logger.debug("STOP whisper recognizer")
        else:
            logger.warning("Recognition is already stop!")
      
    def close(self) -> None:  
        """Close recognition and free resources. Called automatically during __dealloc__
            but if it is necessary then close all manually"""

        if self._is_running:
            logger.info("Close when recognition is run")
            self._is_break = True
            self._is_running = False
            self._audio.pause()
        
        if self._recognize_future is not None and isinstance(self._recognize_future, concurrent.futures.Future) and not self._recognize_future.cancelled():
            res = self._recognize_future.cancel()
            logger.info("Cancel future: %s", res)

        if self._ctx != NULL:  
            whisper_free(self._ctx)  
            logger.info("Free whisper")
        if self._audio != NULL:  
            del self._audio
    
    def start_recognize(self) -> None:
        """Start async recognize"""

        if not self._is_running and self._is_completed:
            # self._recognize_corout = asyncio.to_thread(self.recognize)
            self._recognize_future = asyncio.run_coroutine_threadsafe(asyncio.to_thread(self.recognize), self._loop)
            self._is_running = True
            self._is_break = False  
            self._is_completed = False
            self._audio.resume()
            self.start_time = <int_fast64_t>(round(time.time() * 1000, 3)) # auto-translation to cython types (cpp types)
            logger.debug("Create awaitable recognize future")
        else:
            logger.warning("Recognition is already running!")
    
    async def await_result(self) -> str:
        return await asyncio.wrap_future(self._recognize_future)

    def recognize(self):
        """Sync recognize"""

        cdef vector[whisper_token] prompt_tokens
        self._full_params.prompt_tokens = <whisper_token*>NULL if self._no_context else prompt_tokens.data()  
        self._full_params.prompt_n_tokens = 0 if self._no_context else <int>prompt_tokens.size()

        logger.debug("Start recognize")

        cdef wav_writer wavWriter
        if self.save_audio:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  
            filename = timestamp + ".wav"
            wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1)
        
        fout = None  
        if self.filename_out.length() > 0:  
            try:  
                fout = open(self.filename_out.decode('utf-8'), 'w')  
            except IOError as e:  
                raise RuntimeError(f"Failed to open output file: {e}")

        cdef int n_samples_30s = <int>((1e-3 * self._length_ms) * WHISPER_SAMPLE_RATE)
        cdef vector[float] pcmf32 = vector[float](n_samples_30s, 0.0) 
        cdef int n_iter = 0
        cdef int n_new_line = <int>(max(1.0, float(self._length_ms / self._step_ms - 1))) if not self.use_vad else 1
        cdef int n_samples_keep = <int>((1e-3 * self._keep_ms) * WHISPER_SAMPLE_RATE)

        cdef int n_segment
        cdef int t0
        cdef int t1
        cdef int start_idx 
        cdef int idx
        cdef int j
        cdef int token_count
        cdef const char * text

        cdef int_fast64_t t_last
        cdef int_fast64_t t_now
        cdef int_fast64_t t_diff

        cdef string out_text = "" # return this at the end of function

        t_last = self.start_time
        logger.debug("Came to loop")
        while(True):
            if self._is_break:
                logger.debug("Go out from loop")
                break
            
            t_now = <int_fast64_t>(round(time.time() * 1000, 3))
            t_diff = t_now - t_last
            logger.debug("Time difference: %s", t_diff)
            if (self._is_running and t_diff < self._step_ms):
                logger.debug("Make thread sleep")
                time.sleep(0.1)
                continue
            elif self._is_break:
                continue
            elif not self._is_running:
                logger.debug("Time difference when loop paused: %s", <int>(self.stop_time - t_last))
                self._audio.get(<int>(self.stop_time - t_last), pcmf32)
                self._is_break = True
                t_last = self.stop_time
            else:
                logger.debug("Loop is runnning but the keep time was out.")
                self._audio.get(self._step_ms, pcmf32)
                t_last = t_now
            
            if (self.save_audio):
                wavWriter.write(pcmf32.data(), pcmf32.size())
                logger.debug("Save to audio file")
            
            logger.debug("Came to inference part")

            with nogil:
                res = whisper_full(self._ctx, self._full_params, pcmf32.data(), pcmf32.size())

            if (res != 0):
                raise RuntimeError("Failed to process audio")
            
            # result
            n_segment = whisper_full_n_segments(self._ctx)
            for idx in range(n_segment):
                text = whisper_full_get_segment_text(self._ctx, idx) # get cython's bytes (const char*)

                if self.no_timestamps:
                    out_text += string(text) 
                else:
                    t0 = whisper_full_get_segment_t0(self._ctx, idx)
                    t1 = whisper_full_get_segment_t1(self._ctx, idx)
                    out_text += (  
                        string(b"[") + to_timestamp(t0, False) +   
                        string(b" --> ") + to_timestamp(t1, False) +   
                        string(b"]  ") + string(text) + string(b"\n")  
                    ) 

                if fout is not None:  
                    fout.write(text.decode('utf-8')) # python method must get python's str
            
            if (n_iter % n_new_line) == 0:
                # keep part of the audio for next iteration to try to mitigate word boundary issues
                start_idx = <int>pcmf32.size() - n_samples_keep  
                pcmf32_old = vector[float](pcmf32.begin() + start_idx, pcmf32.end())

                # Add tokens of the last full length segment as the prompt
                prompt_tokens.clear()

                n_segments = whisper_full_n_segments(self._ctx)
                for idx in range(n_segments):
                    token_count = whisper_full_n_tokens(self._ctx, idx)
                    for j in range(token_count):
                        prompt_tokens.push_back(whisper_full_get_token_id(self._ctx, idx, j))
        
            n_iter += 1
            time.sleep(0.1)

        if fout is not None:  
            fout.close()
        self._is_completed = True
        self._audio.pause()
        return out_text.decode('utf-8')
        