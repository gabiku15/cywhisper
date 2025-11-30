from libc.stdint cimport int64_t, int32_t, uint32_t, uint16_t
from libcpp.vector cimport vector
from libcpp.string cimport string

ctypedef bint _bool

cdef nogil:
    cdef const int WHISPER_SAMPLE_RATE = 16000

cdef extern from "ggml.h" nogil:
    enum ggml_log_level:
        GGML_LOG_LEVEL_NONE  = 0,
        GGML_LOG_LEVEL_DEBUG = 1,
        GGML_LOG_LEVEL_INFO  = 2,
        GGML_LOG_LEVEL_WARN  = 3,
        GGML_LOG_LEVEL_ERROR = 4,
        GGML_LOG_LEVEL_CONT  = 5

cdef extern from "whisper.h" nogil:
    enum whisper_sampling_strategy:
        WHISPER_SAMPLING_GREEDY = 0,
        WHISPER_SAMPLING_BEAM_SEARCH
    ctypedef struct whisper_context:
        pass
    ctypedef struct whisper_state:
        pass
    ctypedef struct whisper_ahead:
        int n_text_layer
        int n_head
    ctypedef struct whisper_aheads:
        size_t n_heads
        const whisper_ahead * heads
    ctypedef struct whisper_context_params:  
        _bool  use_gpu  
        _bool  flash_attn  
        int    gpu_device  
        _bool  dtw_token_timestamps  
        int    dtw_aheads_preset  # enum whisper_alignment_heads_preset  
        int    dtw_n_top  
        whisper_aheads dtw_aheads  
        size_t dtw_mem_size
    ctypedef struct whisper_vad_params:
        float threshold
        int min_speech_duration_ms
        int min_silence_duration_ms
        float max_speech_duration_s
        int speech_pad_ms
        float samples_overlap
    ctypedef struct greedy:
        int best_of
    ctypedef struct beam_search:
        int beam_size
        float patience
    ctypedef int whisper_token
    
    ctypedef struct whisper_full_params:
        int strategy
        int n_threads
        int n_max_text_ctx
        int offset_ms
        int duration_ms
        
        _bool translate
        _bool no_context
        _bool no_timestamps
        _bool single_segment
        _bool print_special
        _bool print_progress
        _bool print_realtime
        _bool print_timestamps
        _bool token_timestamps
        _bool tdrz_enable
        
        int max_len
        int max_tokens
        const char* language
        greedy greedy
        beam_search beam_search
        int audio_ctx
        float temperature_inc
        whisper_token* prompt_tokens
        int prompt_n_tokens
        
        _bool vad
        const char * vad_model_path
        whisper_vad_params vad_params

    # logging
    ctypedef void (*ggml_log_callback)(ggml_log_level level, const char * text, void * user_data) noexcept nogil
    cdef void whisper_log_set(ggml_log_callback log_callback, void * user_data)
    
    cdef whisper_context * whisper_init_from_file_with_params(const char* path_model, whisper_context_params params)
    cdef void whisper_free (whisper_context *ctx)
    cdef whisper_full_params whisper_full_default_params (whisper_sampling_strategy strategy)
    cdef int whisper_full(whisper_context* ctx, whisper_full_params params, float* samples, int n_samples)
    cdef int whisper_full_parallel(whisper_context* ctx, whisper_full_params params, const float* samples, int n_samples, int n_processors)
    cdef int whisper_full_n_segments(whisper_context* ctx)
    cdef int64_t whisper_full_get_segment_t0(whisper_context* ctx, int i_segment)
    cdef int64_t whisper_full_get_segment_t1(whisper_context* ctx, int i_segment)
    cdef const char* whisper_full_get_segment_text(whisper_context *ctx, int i_segment)
    cdef int whisper_full_n_segments(whisper_context* ctx)
    cdef int64_t whisper_full_get_segment_t0(whisper_context* ctx, int i_segment)
    cdef int64_t whisper_full_get_segment_t1(whisper_context* ctx, int i_segment)
    cdef whisper_token whisper_full_get_token_id(whisper_context* ctx, int i_segment, int i_token)
    cdef _bool whisper_full_get_segment_speaker_turn_next(whisper_context *ctx, int i_segment)
    cdef int whisper_full_n_tokens (whisper_context * ctx, int i_segment)
    cdef int whisper_lang_id(const char * lang)
    cdef whisper_context_params whisper_context_default_params ()
    cdef whisper_vad_params whisper_vad_default_params()

cdef extern from "common-sdl.h" nogil:
    cdef cppclass audio_async:
        audio_async(int len_ms)
        _bool init(int capture_id, int sample_rate)
        _bool resume()
        _bool pause()
        _bool clear()
        void get(int ms, vector[float]& audio)
    cdef _bool sdl_poll_events()
    ctypedef struct whisper_params:
        int32_t n_threads
        int32_t steps_ms
        int32_t length_ms
        int32_t keep_ms
        int32_t capture_id
        int32_t max_tokens
        int32_t audio_ctx
        int32_t beam_size
        
        float vad_thold
        float freq_thold
        
        _bool translate
        _bool no_fallback
        _bool print_special
        _bool no_context
        _bool no_timestamps
        _bool tinydiarize
        _bool save_audio
        _bool use_gpu
        _bool flash_attn
        
        string language
        string model
        string fname_out

cdef extern from "ggml-backend-reg.cpp" nogil:
    cdef void ggml_backend_load_all()

cdef extern from "common.h" nogil:
    cdef cppclass wav_writer:
        _bool open(const string& filename, const uint32_t sample_rate, const uint16_t bits_per_sample, const uint16_t channels)
        _bool close()
        _bool write(const float *data, size_t length)
    _bool is_file_exist(const char * filename)

cdef extern from "common-whisper.h" nogil:
    string to_timestamp(int64_t t, _bool comma)

    