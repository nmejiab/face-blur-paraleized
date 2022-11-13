// Programa en c++ para difuminar rostros en videos

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

using namespace std;
using namespace cv;

__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

// Function para detectar rostros
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade, double scale);
string cascadeName, nestedCascadeName;

int main(int argc, char* argv[])
{
    ////////////////////////////////////////////////////////////////////////////////////
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
    ////////////////////////////////////////////////////////////////////////////////////

    // Clase VideoCapture para reproducir videos para los cuales se detectarán las caras
    VideoCapture capture;
    //Crea un objeto VideoWriter, aún no inicializado
    VideoWriter writer;
    Mat frame, image;
    string filename;

    // Clasificadores XML entrenados predefinidos con características faciales
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    // Cargar clasificadores
    nestedCascade.load("C:/Users/USUARIO/OpenCV/opencv/build/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    cascade.load("C:/Users/USUARIO/OpenCV/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml");

    // Ruta para el videos
    if (argc != 3) {
        cout << "Video no encontrado" << endl;
    }
    else {
        capture.open(argv[1]);
        filename = argv[2];// Nombre del video de salida
    }

    if (capture.isOpened())
    {
        // capturar fotogramas de vídeo y detectar rostros
        cout << "Deteccion de rostros iniciada" << endl;
        int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        Size frame_size(frame_width, frame_height);
        double fps = capture.get(CAP_PROP_FPS); //after open the capture obj
        int total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);

        cout << "Source info:\n Size:" << frame_size << endl;
        cout << " Frames per seconds:" << fps << endl;
        cout << " Total frames: " << total_frames << endl;

        //Define los fps de video de salida
        int FPS = 30; //Frames per second
        double fontScale = 2;

        //Defina el códec de video por FOURCC, método de grabación, entero fourcc
        int fcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
        //'X','V','I','D' códec de código abierto
        //'M','J','P','G' Vídeo JPEG en movimiento
        //'X','2','6','4' mplementación H.264 de código abierto (comprimido)

        //Inicializar el objeto VideoWriter
        writer = VideoWriter(filename, fcc, FPS, frame_size, true);
        while (1)
        {
            capture >> frame;
            if (frame.empty())
                break;
            Mat frame1 = frame.clone();
            detectAndDraw(frame1, cascade, nestedCascade, scale);
            //Escribe el frame en el archivo de salida.
            writer.write(frame1);
            char c = (char)waitKey(10);

            // Presione q para salir de la ventana
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }
    else
        cout << "Video no encontrado";
    //lanza el video de salida
    writer.release();
    return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade,
    double scale)
{
    vector<Rect> faces, faces2;
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY); // Convierte a scala de grises
    double fx = 1 / scale;

    // Cambiar el tamaño de la imagen en escala de grises
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    // Detecta caras de diferentes tamaños usando el clasificador en cascada 
    cascade.detectMultiScale(smallImg, faces, 1.1,
        2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // Dibuja círculos alrededor de las caras.
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];
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

                // obtener el color promedio del area indicada
                Scalar color = mean(Mat(img, rect));

                // pintar el area indicada con el color obtenido
                rectangle(img, rect, color, cv::FILLED);
            }
        }

        if (nestedCascade.empty())
            continue;
        smallImgROI = smallImg(r);
    }

}