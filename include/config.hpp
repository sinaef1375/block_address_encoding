#pragma once
#include <string> 
struct Config {

    int w_p; //projector width
    int h_p; //projector height
    int w_c; //camera width
    int h_c; //camera height
    int grid_size; //grid size in pixels 
    int eta; //tag size in pixels 
    int n1; //block size
    int K; //Alphabet size
    int input_size; // Classifier input size
    bool second_level; //Perform second level or not
    std::string classifier_directory; // Directory of the traied machine learning model 
    std::string save_result_dir; // Directory to save results in case it is needed
    double threshold; //Threshold
    bool show; // Visualize/Save results or not

};