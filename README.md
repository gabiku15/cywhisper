The Cython binding for Whisper.cpp
The flags is available for compilation according to backend of Whisper.cpp:
    - OPENBLAS=ON/OFF,
    - OPENBLAS_DIR,
    - VULKAN_DIR,
    - VULKAN=ON/OFF,
    - CUDA_DIR,
    - CUDA=ON/OFF,
    (another flags will be add in the future)
Flag with tail part as "_DIR" is the path to main directory of backend's library.
Flag without that is the option, setup.py script check the backend library is available through command-line. Sure, that backend library 
installed on your system and add to PATH variable.
Examples to compile with flags:
OPENBLAS:
    It is recommended to set path to main directory of BLAS library. 
VULKAN:
CUDA: