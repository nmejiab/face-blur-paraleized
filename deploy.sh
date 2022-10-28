FLAGS="$(pkg-config --cflags --libs opencv4)"
g++ -fopenmp parallelFaceBlur.cpp -o parallel $FLAGS
time ./parallel video.mp4 videoExit.mp4 4