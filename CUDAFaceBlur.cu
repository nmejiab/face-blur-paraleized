#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <opencv4/opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

using namespace std;
using namespace cv;

int totalFrames;    
int frameWidth;
int frameHeight;
int numThreads;
int framesPerThread;
int fps;
int fcc;
double scale;
String videoRouteIn;
String videoRouteExit;
string cascadeName, nestedCascadeName;

VideoCapture capture;
Mat **inVideoFrames;
Mat **exitVideoFrames;
Size frameSize;
VideoWriter writer;
CascadeClassifier cascade, nestedCascade;

void setInitialValues(int nArguments, String nameIn, String nameOut, int nThreads);
void setVideoValues();
void fromVideoToMats();
void cloneMats();
void blurVideo();
void writeExitVideo();
void detectAndDraw(Mat& img);

__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

int main(int argc, char* argv[]){
    String nameIn = argv[1];
    String nameOut = argv[2];
    int nThreads = stoi(argv[3]);
    int nArguments = argc;

    setInitialValues(nArguments, nameIn, nameOut, nThreads);
    setVideoValues();
    fromVideoToMats();
    cloneMats();
    blurVideo();
    writeExitVideo();

    for(int i = 0; i < numThreads; i++){
        delete[] inVideoFrames[i];
        delete[] exitVideoFrames[i];
    }

    ////////////////////////////////////////////////////////////////////////////
    int *a, *b, *c;			// host copies of a, b, c
    int *d_a, *d_b, *d_c;		// device copies of a, b, c
    int size = N * sizeof(int);
	
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    ////////////////////////////////////////////////////////////////////////////

    return 0;
}

void setInitialValues(int nArguments, String nameIn, String nameOut, int nThreads){
    if(nArguments == 4){
        videoRouteIn = "/content/" + nameIn;
        videoRouteExit = "/content/" + nameOut;
        numThreads = nThreads;
    }else{
        cout << "not enough arguments";
        exit(0);
    }
}

void setVideoValues(){
    capture.open(videoRouteIn);
    if(!capture.isOpened()){
        cout << "error at open video";
        exit(0);
    }

    double fpsDouble = capture.get(CAP_PROP_FPS);

    frameWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frameHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    frameSize.width = frameWidth;
    frameSize.height = frameHeight;
    totalFrames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    framesPerThread = (int) ceil(totalFrames / numThreads);
    scale = 1;
    fps = fpsDouble;
    fcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
    writer = VideoWriter(videoRouteExit, fcc, fps, frameSize, true);
    nestedCascade.load("/home/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    cascade.load("/home/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
}

void fromVideoToMats(){
    inVideoFrames = new Mat*[numThreads];
    for(int i = 0; i < numThreads; i++){
        inVideoFrames[i] = new Mat[framesPerThread * 2];
    }
    for(int i = 0; i < numThreads; i++){
        for(int j = 0; j < framesPerThread; j++){
            capture >> *(*(inVideoFrames + i) + j);
            Mat evaluate = *(*(inVideoFrames + i) + j);
            if(evaluate.empty())
                break;
        }
    }
}

void cloneMats(){
    exitVideoFrames = new Mat*[numThreads];
    for(int i = 0; i < numThreads; i++){
        exitVideoFrames[i] = new Mat[framesPerThread * 2];
    }
    omp_set_num_threads(numThreads);
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for(int i = 0; i < framesPerThread; i++){
            Mat evaluate = *(*(inVideoFrames + id) + i);
            if(evaluate.empty())
                continue;
            *(*(exitVideoFrames + id) + i) = evaluate;
        }
    }
}

void blurVideo(){
    omp_set_num_threads(numThreads);
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for(int i = 0; i < framesPerThread;i++){
            Mat evaluate = *(*(exitVideoFrames + id) + i);
            if(evaluate.empty())
                continue;
            detectAndDraw(*(*(exitVideoFrames + id) + i));
        }
    }
}

void writeExitVideo(){
    for(int i = 0; i < numThreads; i++){
        for(int j = 0; j < framesPerThread; j++){
            Mat evaluate = *(*(exitVideoFrames + i) + j);
            if(evaluate.empty())
                continue;
            writer.write(*(*(exitVideoFrames + i) + j));
        }
    }
}

void detectAndDraw(Mat& img){
    vector<Rect> faces;
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY); // transform to grayscale
    double fx = 1 / scale;

    // resize the grayscale image
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    // detect faces of different sizes using the cascade classifier 
    cascade.detectMultiScale(smallImg, faces, 1.1,
        2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // draw circles around the faces
    for (size_t h = 0; h < faces.size(); h++)
    {
        Rect r = faces[h];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        
        int pixel_size = 16;
        Rect rect;
        for (int i = 0; i < r.width; i += pixel_size)
        {
            for (int j = 0; j < r.height; j += pixel_size)
            {
                rect.x = r.x + j;
                rect.y = r.y + i;
                rect.width = j + pixel_size < r.height ? pixel_size : r.height - j;
                rect.height = i + pixel_size < r.width ? pixel_size : r.width - i;

                // get the average color of the indicated area
                Scalar color = mean(Mat(img, rect));

                // paint the indicated area with the obtained color
                rectangle(img, rect, color, cv::FILLED);
            }
        }

        if (nestedCascade.empty())
            continue;
        smallImgROI = smallImg(r);
    }
}