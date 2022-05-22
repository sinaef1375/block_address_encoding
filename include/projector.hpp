#pragma once 
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class Projector {
public:
	int width; 
	int height;
	Eigen::Matrix3d calibration_matrix; 

	Projector();
	Projector(int w, int h);
	Projector(int w, int h, Eigen::Matrix3d k);

};