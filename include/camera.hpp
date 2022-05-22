#pragma once 
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class Camera {
public:
	int width;
	int height;
	Eigen::Matrix3d calibration_matrix;

	Camera();
	Camera(int w, int h);
	Camera(int w, int h, Eigen::Matrix3d k);

};