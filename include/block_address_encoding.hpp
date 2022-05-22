#pragma once
#include <include/config.hpp>
#include <include/camera.hpp>
#include <include/projector.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
using namespace cv;
using namespace std;
using namespace Eigen;

namespace blockAddressEncoding {

    int patternGeneration(Config config) {
        return 0;
    }

    void largestConnecttedComponent(const Mat& mask)
    {
        std::shared_ptr<Mat> labels = std::make_shared<Mat>();

        //1. Mark connected domain
        int n_comps = connectedComponents(mask, *labels, 8, CV_16U);
        vector<int> histogram_of_labels;
        for (int i = 0; i < n_comps; i++) //The number of initialized labels is 0
        {
            histogram_of_labels.push_back(0);
        }

        int rows = labels->rows;
        int cols = labels->cols;
        for (int row = 0; row < rows; row++) //Calculate the number of each label
        {
            for (int col = 0; col < cols; col++)
            {
                histogram_of_labels.at(labels->at<unsigned short>(row, col)) += 1;
            }
        }
        histogram_of_labels.at(0) = 0; //Set the number of background labels to 0

        //2. Calculate the largest connected domain labels index
        int maximum = 0;
        int max_idx = 0;
        for (int i = 0; i < n_comps; i++)
        {
            if (histogram_of_labels.at(i) > maximum)
            {
                maximum = histogram_of_labels.at(i);
                max_idx = i;
            }
        }

        //3. Mark the maximum connected domain as 1
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                if (labels->at<unsigned short>(row, col) == max_idx)
                {
                    labels->at<unsigned short>(row, col) = 255;
                }
                else
                {
                    labels->at<unsigned short>(row, col) = 0;
                }
            }
        }
        //4. Change the image to CV_8U format
        labels->convertTo(mask, CV_8U);
    }

    // Tag detection and finding the bound box and centroid of each tag 

    void tagDetection(const Config& config, const Mat& I_C, const std::shared_ptr<Mat> bb, const std::shared_ptr<Mat> centroids) {

        // Mask out the tags 
        std::shared_ptr<Mat> mask = std::make_shared<Mat>();
        threshold(I_C, *mask, config.threshold, 255, 0);
        largestConnecttedComponent(*mask);

        floodFill(*mask, Point(0, 0), Scalar(255));
        bitwise_not(*mask, *mask);
        if (config.show)
        {
            namedWindow("Mask", WINDOW_NORMAL);
            imshow("Mask", *mask);
            cv::imwrite(config.save_result_dir +"/mask.png", *mask);
        }

        // obtain the connected components in the mask image
        Mat labelImage;
        int n_labels = connectedComponentsWithStats(*mask, labelImage, *bb, *centroids, 8, CV_32S);	 

        // Eliminating connected components wiht less than 1000 pixels and more than 100000 pixels 
        std::shared_ptr<Mat> stats_refined = std::make_shared<Mat>(); // Refined centroids after elemination of invalid components 
        std::shared_ptr<Mat> centroids_refined = std::make_shared<Mat>(); // Refined bounding boxes after elemination of invalid components

        int count = 0;
        for (int i = 1; i < bb->rows; i++) {
            if (bb->at<int>(i, bb->cols - 1) > 100 && bb->at<int>(i, bb->cols - 1) < 100000) {
                count = count + 1;
                if (count == 1)
                {
                    *stats_refined = (*bb)(Rect(0, i, bb->cols, 1));
                    *centroids_refined = (*centroids)(Rect(0, i, centroids->cols, 1));
                }
                else
                {
                    vconcat(*stats_refined, (*bb)(Rect(0, i, bb->cols, 1)), *stats_refined);
                    vconcat(*centroids_refined, (*centroids)(Rect(0, i, centroids->cols, 1)), *centroids_refined);
                }
            }
        }
        *bb = (*stats_refined)(Rect(0, 0, stats_refined->cols - 1, stats_refined->rows)); // Bounding boxes   
        *centroids = (*centroids_refined)(Rect(0, 0, centroids_refined->cols, centroids_refined->rows)); // Centroids   
    }

    

    void find_loc(const Mat& matrix, const int val, Mat& idx) {

        int count = 0;
        for (int i = 0; i < matrix.rows; i++) {
            if (matrix.at<int>(i, 0) == val)
            {
                count = count + 1;
                if (count == 1)
                {
                    idx = cv::Mat(1, 1, CV_32S, cv::Scalar(i));
                }
                else
                {
                    vconcat(idx, cv::Mat(1, 1, CV_32S, cv::Scalar(i)), idx);
                }
            }
        }
    }

    // Tag classfication (inference)

    void tag_classification(const Mat& bb, const Ptr<ml::SVM>& model, const Mat& I_C, const Config& config, Mat& predicted_labels) {
        
        int img_area = config.input_size * config.input_size;                           // The size of the classifier input 
        std::shared_ptr<Mat> data = std::make_shared<Mat>(bb.rows, img_area, CV_32FC1); // Test data 
        std::shared_ptr<Mat> single_bb = std::make_shared<Mat>();                       // A single bounding box 
        std::shared_ptr<Mat> bb_image = std::make_shared<Mat>();                        // A single bounding box image 
        
        // Create the test data 
        for (int i = 0; i < bb.rows; i++) {

            *single_bb = bb(Rect(0, i, bb.cols, 1));
            *bb_image = I_C(Rect(single_bb->at<int>(0, 0), single_bb->at<int>(0, 1), single_bb->at<int>(0, 2), single_bb->at<int>(0, 3)));
            resize(*bb_image, *bb_image, Size(config.input_size, config.input_size));

            int ii = 0;                                 
            for (int iii = 0; iii < bb_image->rows; iii++)
            {
                for (int jjj = 0; jjj < bb_image->cols; jjj++)
                {
                    data->at<float>(i, ii) = bb_image->at<uchar>(iii, jjj);  
                    ii++;
                }
            }

        }
        //Predict labels 
        model->predict(*data, predicted_labels);
    }

    /////////////////////////////////// Delete row or column of a martix 

    Mat mat_remove(Mat Matrix, int index, string config) {

        Mat Matrix_modified;
        if (config == "row") {
            if (index > 0 && index < Matrix.rows - 1) {
                Mat b, c;
                Matrix(Rect(0, 0, Matrix.cols, index)).copyTo(b);
                Matrix(Rect(0, index + 1, Matrix.cols, Matrix.rows - index - 1)).copyTo(c);
                vconcat(b, c, Matrix_modified);
            }
            else if (index == 0) {
                Matrix_modified = Matrix(Rect(0, 1, Matrix.cols, Matrix.rows - 1));
            }
            else if (index == Matrix.rows - 1) {
                Matrix_modified = Matrix(Rect(0, 0, Matrix.cols, Matrix.rows - 1));
            }
        }
        if (config == "column") {
            if (index > 0 && index < Matrix.cols - 1) {
                Mat b, c;
                Matrix(Rect(0, 0, index, Matrix.rows)).copyTo(b);
                Matrix(Rect(index + 1, 0, Matrix.cols - index - 1, Matrix.rows)).copyTo(c);
                hconcat(b, c, Matrix_modified);
            }
            else if (index == 0) {
                Matrix_modified = Matrix(Rect(1, 0, Matrix.cols - 1, Matrix.rows));
            }
            else if (index == Matrix.cols - 1) {
                Matrix_modified = Matrix(Rect(0, 0, Matrix.cols - 1, Matrix.rows));
            }
        }
        return Matrix_modified;
    }
    //////////////////////////////////// Delete row or column of a martix

    void mat_remove(const Mat& mat, const Mat& idx, const bool removeRows, Mat& result)
    {
        cv::sort(idx, idx, SORT_EVERY_COLUMN + SORT_ASCENDING);
        std::shared_ptr<Mat> res = std::make_shared<Mat>();
        int count = 0;
        int n = removeRows ? mat.rows : mat.cols;
        for (int i = 0; i < n; i++)
        {
            if (idx.at<int>(count, 0) != i) {
                Mat rc = removeRows ? mat.row(i) : mat.col(i);
                if ((*res).empty()) *res = rc;
                else
                {
                    if (removeRows)
                        vconcat(*res, rc, *res);
                    else
                        hconcat(*res, rc, *res);
                }
            }
            else {
                count++;
            }
        }
        result = *res;
    }

    //////////////////////////////////// Slice a set of rows/columns of a matrix  

    void matSlice(const Mat& mat, const Mat& idx, const bool sliceRows, Mat& result)
    {
        cv::sort(idx, idx, SORT_EVERY_COLUMN + SORT_ASCENDING);
        std::shared_ptr<Mat> res = std::make_shared<Mat>();
        int count = 0;
        int n = sliceRows ? mat.rows : mat.cols;
        for (int i = 0; i < idx.rows; i++) {
            Mat rc = sliceRows ? mat.row(idx.at<int>(i, 0)) : mat.col(idx.at<int>(i, 0));
            if ((*res).empty()) *res = rc;
            else {
                if (sliceRows) {
                    vconcat(*res, rc, *res);
                }
                else {
                    hconcat(*res, rc, *res);
                }

            }
        }
        /*
        for (int i = 0; i < n; i++)
        {
            if (idx.at<int>(count, 0) == i) {
                Mat rc = sliceRows ? mat.row(i) : mat.col(i);
                if ((*res).empty()) *res = rc;
                else
                {
                    if (sliceRows)
                        vconcat(*res, rc, *res);
                    else
                        hconcat(*res, rc, *res);
                }
                count++;
            }
        }*/
        result = *res;
    }

    /////////////////////////////////// find the overlap ratios ////////////////////////////////////

    Mat overlapratio(Mat BB, Rect box) {
        Mat O(1, 1, CV_32SC4, Scalar(0)), Idx1(1, 1, CV_32SC4, Scalar(0));
        for (int i = 0; i < BB.rows; i++) {
            Rect box1(BB.at<int>(i, 0), BB.at<int>(i, 1), BB.at<int>(i, 2), BB.at<int>(i, 3));
            int intersects = ((box1 & box).area() > 0);
            if (i == 0) {
                O.at<int>(i, 0) = intersects;
            }
            else
            {
                Idx1.at<int>(0, 0) = intersects;
                vconcat(O, Idx1, O);
            }
        }
        return O;
    }

    ///////////////// Distance between rows of two matrices (m*n)

    Mat mat_distance(Mat Matrix1, Mat Matrix2) {

        Mat distance = (Mat_<double>(1, 1) << 0);
        for (int i = 0; i < Matrix1.rows; i++) {
            if (i == 0) {
                distance.at<double>(0, 0) = norm(Matrix1(Rect(0, i, Matrix1.cols, 1)) - Matrix2(Rect(0, i, Matrix2.cols, 1)));
            }
            else {
                vconcat(distance, norm(Matrix1(Rect(0, i, Matrix1.cols, 1)) - Matrix2(Rect(0, i, Matrix2.cols, 1))), distance);
            }
        }
        return distance;
    }

    ///////////////// Block detection (Obtaning block codewords) with tag indices

    auto block_detection(const Mat& bb, const Mat& marker_indices, const int n1, const Mat& labels, Mat& tag_indices, Mat& block_codewords) {

      
        tag_indices = cv::Mat::zeros(1, n1 * n1, CV_32S);
        block_codewords = cv::Mat::zeros(1, n1 * n1 - 1, CV_32S);
        Mat O, C11, C12, C21, C22, neighbor_index, index_loc, top_left, top_right, bottom_left, bottom_right, C1, C2, C3, C4, d, C_repeat;;
        int count = 0;
        for (int i = 0; i < marker_indices.rows; i++)
        {
            // find the bounding boxes in the neighborhood of each marker tag 
            int index = marker_indices.at<int>(i, 0);
            Rect box(bb.at<int>(index, 0) - 4 * bb.at<int>(index, 2), bb.at<int>(index, 1) - 4 * bb.at<int>(index, 3), bb.at<int>(index, 2) + 8 * bb.at<int>(index, 2), bb.at<int>(index, 3) + 8 * bb.at<int>(index, 3));
            O = overlapratio(bb, box);
            int idx_number = sum(O)[0];
            if (idx_number > n1 * n1) {
                neighbor_index;
                find_loc(O, 1, neighbor_index);
                // Obtaining the corners of the marker tag
                C11 = bb(Rect(0, index, 2, 1));           
                C11.copyTo(C12);
                C11.copyTo(C21);
                C11.copyTo(C22);

                C12.at<int>(0, 0) = (C11.at<int>(0, 0) + bb.at<int>(index, 2));
                C21.at<int>(0, 1) = (C11.at<int>(0, 1) + bb.at<int>(index, 3));
                C22.at<int>(0, 0) = (C11.at<int>(0, 0) + bb.at<int>(index, 2));
                C22.at<int>(0, 1) = (C11.at<int>(0, 1) + bb.at<int>(index, 3));

                find_loc(neighbor_index, index, index_loc);
                neighbor_index = mat_remove(neighbor_index, index_loc.at<int>(0,0), "row");

                /// Obtaining the coreners of all tags in the neighbirhood 
                for (int i = 0; i < neighbor_index.rows - 1; i++) {
                    C1 = bb(Rect(0, neighbor_index.at<int>(i, 0), 2, 1));
                    C1.copyTo(C2);
                    C1.copyTo(C3);
                    C1.copyTo(C4);

                    C2.at<int>(0, 0) = C1.at<int>(0, 0) + bb.at<int>(neighbor_index.at<int>(i, 0), 2);
                    C3.at<int>(0, 1) = C1.at<int>(0, 1) + bb.at<int>(neighbor_index.at<int>(i, 0), 3);
                    C4.at<int>(0, 0) = C1.at<int>(0, 0) + bb.at<int>(neighbor_index.at<int>(i, 0), 2);
                    C4.at<int>(0, 1) = C1.at<int>(0, 1) + bb.at<int>(neighbor_index.at<int>(i, 0), 3);
                    if (i == 0) {
                        top_left = C1;
                        top_right = C2;
                        bottom_left = C3;
                        bottom_right = C4;
                    }
                    else {
                        vconcat(top_left, C1, top_left);
                        vconcat(top_right, C2, top_right);
                        vconcat(bottom_left, C3, bottom_left);
                        vconcat(bottom_right, C4, bottom_right);
                    }
                }

                /// Obtaining block code-word
                
                // Tag 1
                repeat(C11, bottom_right.rows, 1, C_repeat);
                Mat d = mat_distance(C_repeat, bottom_right);
                double minVal, maxVal;
                Point  minLoc, maxLoc;
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 0) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 0) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    vconcat(tag_indices, cv::Mat::zeros(1, n1 * n1, CV_32S), tag_indices);
                    vconcat(block_codewords, cv::Mat::zeros(1, n1 * n1 - 1, CV_32S), block_codewords);
                    tag_indices.at<int>(count, 0) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 0) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }

                // Tag 2

                repeat(C21, bottom_right.rows, 1, C_repeat);
                d = mat_distance(C_repeat, bottom_right);
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 1) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 1) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    tag_indices.at<int>(count, 1) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 1) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }

                // Tag 3

                repeat(C21, top_right.rows, 1, C_repeat);
                d = mat_distance(C_repeat, top_right);
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 2) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 2) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    tag_indices.at<int>(count, 2) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 2) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }

                // Tag 4 
                repeat(C11, bottom_left.rows, 1, C_repeat);
                d = mat_distance(C_repeat, bottom_left);
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 3) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 3) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    tag_indices.at<int>(count, 3) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 3) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }

                // Tag 5 (middle)
                if (count == 0) {
                    tag_indices.at<int>(count, 4) = index;
                }
                else {
                    tag_indices.at<int>(count, 4) = index;
                }

                // Tag 6
                repeat(C21, top_left.rows, 1, C_repeat);
                d = mat_distance(C_repeat, top_left);
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 5) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 4) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    tag_indices.at<int>(count, 5) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 4) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }

                // Tag 7
                repeat(C12, bottom_left.rows, 1, C_repeat);
                d = mat_distance(C_repeat, bottom_left);
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 6) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 5) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    tag_indices.at<int>(count, 6) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 5) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }

                // Tag 8
                repeat(C12, top_left.rows, 1, C_repeat);
                d = mat_distance(C_repeat, top_left);
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 7) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 6) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    tag_indices.at<int>(count, 7) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 6) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }

                // Tag 9          
                repeat(C22, top_left.rows, 1, C_repeat);
                d = mat_distance(C_repeat, top_left);
                minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);

                if (count == 0) {
                    tag_indices.at<int>(count, 8) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 7) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                else {
                    tag_indices.at<int>(count, 8) = neighbor_index.at<int>(minLoc.y, minLoc.x);
                    block_codewords.at<int>(count, 7) = labels.at<int>(neighbor_index.at<int>(minLoc.y, minLoc.x), 0);
                    neighbor_index = mat_remove(neighbor_index, minLoc.y, "row");
                    top_left = mat_remove(top_left, minLoc.y, "row");
                    top_right = mat_remove(top_right, minLoc.y, "row");
                    bottom_left = mat_remove(bottom_left, minLoc.y, "row");
                    bottom_right = mat_remove(bottom_right, minLoc.y, "row");
                }
                count++;
            }
        }
    }

    //////////////////////////////////////////////// Find the pixel correspondences 

    void findCorrespondences(const Config& config, const Mat& I_P, const Mat& I_C, const Ptr<ml::SVM> model, Mat& xy_cam, Mat& uv_pro) {

        // Tag detection  
        std::shared_ptr<Mat> bb = std::make_shared<Mat>(); // bounding boxes of the detected tags 
        std::shared_ptr<Mat> centroids = std::make_shared<Mat>(); // centroids of the detected tags 
        tagDetection(config, I_C, bb, centroids); // (obtaining bounding boxes + centroids)
       
        // Tag classification 
        std::shared_ptr<Mat> predicted_labels = std::make_shared<Mat>();
        std::shared_ptr<Mat> marker_indices = std::make_shared<Mat>();
        tag_classification(*bb, model, I_C, config, *predicted_labels);
        predicted_labels->convertTo(*predicted_labels, CV_32SC4);
        find_loc(*predicted_labels, config.K, *marker_indices); // FInd the index of marker tags (middle tags)

        // Block detection
        std::shared_ptr<Mat> tag_indices = std::make_shared<Mat>(); // Indices of tags in a block 
        std::shared_ptr<Mat> block_codewords = std::make_shared<Mat>(); // Block codewords 
        block_detection(*bb, *marker_indices, config.n1, *predicted_labels, *tag_indices, *block_codewords);

        // Error Detection
        Mat rows_with_error = (((*block_codewords)(Rect(0, 0, block_codewords->cols/4, block_codewords->rows))) == ((*block_codewords)(Rect(2, 0, block_codewords->cols/4, block_codewords->rows))) );
        Mat columns_with_error = (((*block_codewords)(Rect(4, 0, block_codewords->cols/4, block_codewords->rows))) == ((*block_codewords)(Rect(6, 0, block_codewords->cols / 4, block_codewords->rows))));
        bitwise_and(rows_with_error(Rect(0, 0, 1, rows_with_error.rows)), rows_with_error(Rect(1, 0, 1, rows_with_error.rows)), rows_with_error);
        bitwise_and(columns_with_error(Rect(0, 0, 1, columns_with_error.rows)), columns_with_error(Rect(1, 0, 1, columns_with_error.rows)), columns_with_error);
        Mat blocks_with_error, error_idx; 
        bitwise_and(columns_with_error, columns_with_error, error_idx);
        bitwise_not(error_idx, error_idx);
        for (int i = 0; i<error_idx.rows; i++) {
            if (error_idx.at<uchar>(i, 0) == 255) {
                if (blocks_with_error.empty()) {
                    blocks_with_error = cv::Mat(1, 1, CV_32S, cv::Scalar(i));
                }
                else{
                    vconcat(blocks_with_error, cv::Mat(1,1, CV_32S, cv::Scalar(i)), blocks_with_error);
                }
            }
        }
        
        mat_remove(*block_codewords, blocks_with_error, 1,*block_codewords);
        mat_remove(*tag_indices, blocks_with_error, 1, *tag_indices);
        Mat rows = (*block_codewords)(Rect(0, 0, (config.n1*config.n1-1)/4, block_codewords->rows));
        Mat cols = (*block_codewords)(Rect((config.n1*config.n1 -1)/2, 0, (config.n1*config.n1-1)/4, block_codewords->rows));
        rows = rows(Rect(0, 0, 1, rows.rows))*(config.K) + rows(Rect(1, 0, 1, rows.rows));
        cols = cols(Rect(0, 0, 1, cols.rows))*(config.K) + cols(Rect(1, 0, 1, cols.rows));
        Mat uv_pro_temp = Mat();
        for (int i = 0; i<config.n1; i++) {
                for (int j = 0; j<config.n1; j++) {
                    hconcat( (cols)*(config.grid_size+config.eta)*config.n1+(j+0.5)*(config.grid_size + config.eta), 
                             (rows) * (config.grid_size + config.eta) * config.n1 + (i+0.5)*(config.grid_size + config.eta), uv_pro_temp);
                    if (i == 0 && j == 0) {
                        uv_pro = uv_pro_temp;
                    }
                    else {
                        vconcat(uv_pro, uv_pro_temp, uv_pro);
                    }
                }
        }
        
        *tag_indices = tag_indices->reshape(0, tag_indices->cols * tag_indices->rows);
        matSlice(*centroids,*tag_indices, 1, xy_cam);

    };

}