#include <include/projector.hpp>
#include <include/camera.hpp>
#include <include/config.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <include/block_address_encoding.hpp>
using namespace cv;
using namespace Eigen;
using namespace std;

int main() {

    ////// Setting experiment configuration //////
	Config config;
    config.w_p = 1280;                           //projector width
    config.h_p = 800;                            //projector height
    config.w_c = 2448;                           //camera width
    config.h_c = 2048;                           //camera height
    config.grid_size = 4;                        //grid size in pixels 
    config.eta = 20;                             //tag size in pixels 
    config.n1 = 3;                               //block size
    config.K = 6;                                //Alphabet size
    config.input_size = 50;                      //Classifier input size
    config.second_level = 0;                     //Perform second level or not
    config.classifier_directory = "C:/Users/sfars/Desktop/courses/C++/address_encoding/output.xml";  //Directory ML model 
    config.save_result_dir = "C:/Users/sfars/Desktop/new_folder/applications/Job applications/DarkVision";
    config.threshold = 45;                       //Threshold value
    config.show = 0;                             //Visualize data or not

    ///// Initializing camera and projector objects //////
    Camera c1;
    Projector p1;

    // Creating structured light image
    std::shared_ptr<Mat> I_P = std::make_shared<Mat>(); 
    *I_P = imread("C:/Users/sfars/Desktop/courses/C++/address_encoding/Structured_Light_Image.png");

    if (!(*I_P).data)
    {
        printf("No image data \n");
        return -1;
    }
    if (config.show)
    {
        namedWindow("I_P", WINDOW_NORMAL);
        imshow("I_P", *I_P);
    }

    // Loading captured camera image 
    std::shared_ptr<Mat> I_C = std::make_shared<Mat>();
    *I_C = imread("C:/Users/sfars/Desktop/courses/C++/address_encoding/Camera_Image.png", IMREAD_GRAYSCALE);
    if (!(*I_C).data)
    {
        printf("No image data \n");
        return -1;
    }
    if (config.show)
    {
        namedWindow("I_C", WINDOW_NORMAL);
        imshow("I_C", *I_C);
        //imwrite("C:/Users/sfars/Desktop/a.png", *I_C);
    }

    // Tag classifier
    Ptr<ml::SVM> model;
    model = Algorithm::load<ml::SVM>(config.classifier_directory);  //Loading the trained classifier
   
    // Obtaining pixel correspondences  
    std::shared_ptr<Mat> xy_cam = std::make_shared<Mat>(); 
    std::shared_ptr<Mat> uv_pro = std::make_shared<Mat>();
    blockAddressEncoding::findCorrespondences(config, *I_P, *I_C, model, *xy_cam, *uv_pro);

	return 0;
}