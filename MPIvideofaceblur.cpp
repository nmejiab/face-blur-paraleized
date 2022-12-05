// Programa en c++ para difuminar rostros en videos

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <mpi.h>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512
#define MSG_LENGTH 2

using namespace std;
using namespace cv;
using namespace MPI;

void Add(const float *A, const float *B, float *C, int numElements, int pixel_size)
{
    int i, tag = 1, nodes, iam, back[2], recieve[2 * nodes];
    MPI_Status status;
    
    MPI_Init();
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);

    int cant = (int)ceil(numElements / nodes);
    float message[MSG_LENGTH * cant];

    if (iam == 0) {
        for (i = 1; i < nodes; i++){
            for(int j = 0; j < cant; j++){
                if(i * cant + j > numElements){
                    message[j * 2] = -1;
                    message[j * 2 + 1] = -1;
                }
                message[j * 2] = A[i * cant + j];
                message[j * 2 + 1] = B[i * cant + j];
            }
            Comm::Send(message, MSG_LENGTH * cant, MPI_FLOAT, i, tag, 
                    MPI_COMM_WORLD);
        }
    } else {

        Comm::Recv(message, MSG_LENGTH * cant, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, 
                &status);

    }
    if(message[0] >= 0 && message[1] >= 0){
        if(iam < 2){
            back[0] = message[0] + message[1];
        }else{
            back[0] = message[1] + pixel_size < message[0] ? 
                    pixel_size : message[0] - message[1];
        }
    }else back[0] = -1;

    back[1] = iam;

    MPI_Gather(back, 2, MPI_INT, recieve, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (iam == 0) {
        c[0] = back[0];
        for(i = 2; i < 8; i += 2){
            c[i / 2] = recieve[i];
        }
    }

    MPI_Finalize();
}

// Function para detectar rostros
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade, double scale, int threadsPerBlock);
string cascadeName, nestedCascadeName;

int main(int argc, char* argv[])
{

    // Clase VideoCapture para reproducir videos para los cuales se detectarán las caras
    VideoCapture capture;
    //Crea un objeto VideoWriter, aún no inicializado
    VideoWriter writer;
    Mat frame, image;
    string filename;
    int threadsPerBlock;
    // Clasificadores XML entrenados predefinidos con características faciales
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    // Cargar clasificadores
    nestedCascade.load("/home/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    cascade.load("/home/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
    
       
    
    if (argc != 4) {
        cout << "Error en numero de parametros de entrada" << endl;
        exit(0);
    }
    else {
        // Ruta para el videos
        String nameIn = argv[1];
        String nameOut = argv[2];
        threadsPerBlock = stoi(argv[3]);
        String videoRouteIn = "/content/" + nameIn;
        String videoRouteExit = "/content/" + nameOut;
        capture.open(videoRouteIn);
        filename = videoRouteExit;// Nombre del video de salida
    }
    int total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    if (capture.isOpened())
    {
        // capturar fotogramas de vídeo y detectar rostros
        cout << "Deteccion de rostros iniciada" << endl;
        int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        Size frame_size(frame_width, frame_height);
        double fps = capture.get(CAP_PROP_FPS); //after open the capture obj
        int total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);

        //Define los fps de video de salida
        int FPS = fps; //Frames per second

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
            if (frame.empty()){
                break;
            }
            Mat frame1 = frame.clone();
            detectAndDraw(frame1, cascade, nestedCascade, scale, threadsPerBlock);
            //Escribe el frame en el archivo de salida.
            writer.write(frame1);
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
    double scale, int threadsPerBlock)
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
    // Pixela las caras.
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
                // Print the vector length to be used, and compute its size
                int numElements = 4;
                size_t size = numElements * sizeof(float);

                // Allocate the host input vector A
                float *h_A = (float *)malloc(size);

                // Allocate the host input vector B
                float *h_B = (float *)malloc(size);

                // Allocate the host output vector C
                float *h_C = (float *)malloc(size);

                h_A[0] = r.x;
                h_B[0] = j;
                h_A[1] = r.y;
                h_B[1] = i;
                h_A[2] = r.height;
                h_B[2] = j;
                h_A[3] = r.width;
                h_B[3] = i;
                
                Add(h_A, h_B, h_C, numElements, pixel_size);

                rect.x = h_C[0];
                rect.y = h_C[1];
                rect.width = h_C[2];
                rect.height = h_C[3];
                
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
