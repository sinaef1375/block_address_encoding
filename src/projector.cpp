#include "include\projector.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace Eigen;

Projector::Projector() : width{ 1920 }, height{ 1080 }, calibration_matrix{Matrix3d::Identity(3,3)} {};

Projector::Projector(int w, int h) : width{w}, height{h}, calibration_matrix{ Matrix3d::Identity(3,3) } {};

Projector::Projector(int w, int h, Matrix3d k) : width{w}, height{h}, calibration_matrix{k} {};

