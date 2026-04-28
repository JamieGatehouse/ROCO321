#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "../owl.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    //--------------------- INITIALISE ROBOT ---------------------
    // Create robotOwl object with calibration values
    robotOwl owl(1500, 1505, 1500, 1590, 1520);

    //--------------------- LOAD CALIBRATION FILES ---------------------
    // File paths for intrinsic and extrinsic calibration
    string intrinsic_filename = "../intrinsics.xml";
    string extrinsic_filename = "../extrinsics.xml";

    // Rectangles for valid image regions after rectification
    Rect roi1, roi2;

    // Q matrix used for 3D reconstruction
    Mat Q;

    // Image size (must match calibration resolution)
    Size img_size = {640, 480};

    // Open intrinsic calibration file
    FileStorage fs(intrinsic_filename, FileStorage::READ);
    if (!fs.isOpened()) {
        printf("Failed to open file %s\n", intrinsic_filename.c_str());
        return -1;
    }

    // Camera matrices (M) and distortion coefficients (D)
    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    // Open extrinsic calibration file
    fs.open(extrinsic_filename, FileStorage::READ);
    if (!fs.isOpened()) {
        printf("Failed to open file %s\n", extrinsic_filename.c_str());
        return -1;
    }

    // Rotation (R) and translation (T) between cameras
    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    //--------------------- STEREO RECTIFICATION ---------------------
    // Align both camera images so corresponding points lie on same row
    stereoRectify(M1, D1, M2, D2, img_size, R, T,
                  R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

    // Precompute remapping functions for distortion correction
    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    //--------------------- DISTANCE CALIBRATION ---------------------
    // Known real-world distances (in cm)
    vector<float> known_distances = {50.0, 100.0, 150.0};

    // Corresponding disparity values from stereo system
    vector<float> known_disparities = {100.0, 50.0, 25.0};

    // Constant used in distance calculation: distance = C / disparity
    float C = 0;

    // Compute calibration constant
    for (size_t i = 0; i < known_distances.size(); ++i) {
        C = known_disparities[i] * known_distances[i];
    }

    //--------------------- CREATE STEREO MATCHER ---------------------
    // Parameters for StereoSGBM algorithm
    int SADWindowSize = 5;          // Block size (must be odd)
    int numberOfDisparities = 256;  // Must be divisible by 16

    // Create StereoSGBM object
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

    // Set matching parameters
    sgbm->setBlockSize(SADWindowSize);
    sgbm->setPreFilterCap(63);
    sgbm->setP1(8 * 3 * SADWindowSize * SADWindowSize);
    sgbm->setP2(32 * 3 * SADWindowSize * SADWindowSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);

    //--------------------- MAIN LOOP ---------------------
    // Continuously capture and process frames
    while (1) {
        // Matrices for left and right camera images
        Mat left, right;

        // Capture stereo frames
        owl.getCameraFrames(left, right);

        // Check if frames were captured successfully
        if (left.empty() || right.empty()) {
            cerr << "Failed to capture image frames!" << endl;
            break;
        }

        //--------------------- IMAGE RECTIFICATION ---------------------
        // Apply remapping to correct distortion and align images
        remap(left, left, map11, map12, INTER_LINEAR);
        remap(right, right, map21, map22, INTER_LINEAR);

        //--------------------- DISPARITY COMPUTATION ---------------------
        // Compute disparity map from stereo pair
        Mat disp, disp8;
        sgbm->compute(left, right, disp);

        // Convert disparity to 8-bit image for visualisation only
        disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * 16.));

        //--------------------- DISTANCE MAP CREATION ---------------------
        // Create floating-point matrix to store distance values
        Mat distance_map = Mat::zeros(disp.size(), CV_32F);

        // Loop through every pixel
        for (int y = 0; y < disp.rows; ++y) {
            for (int x = 0; x < disp.cols; ++x) {

                // Ignore invalid disparity values
                if (disp.at<short>(y, x) > 0) {

                    // Convert disparity from fixed-point format
                    float disparity = disp.at<short>(y, x) / 16.0f;

                    // Calculate distance using calibrated constant
                    float distance = C / disparity;

                    // Store distance in map
                    distance_map.at<float>(y, x) = distance;
                }
            }
        }

        //--------------------- REGION OF INTEREST ---------------------
        // Define a central region for demonstration
        Rect boundingBox(left.cols / 3, left.rows / 3,
                         left.cols / 3, left.rows / 3);

        // Draw bounding box on left image
        rectangle(left, boundingBox, Scalar(0, 255, 0), 2);

        //--------------------- DISTANCE AT CENTRE ---------------------
        // Compute centre point of image
        Point center(left.cols / 2, left.rows / 2);

        // Retrieve distance value at centre
        float center_distance = distance_map.at<float>(center);

        //--------------------- DISPLAY DISTANCE ---------------------
        // Convert distance to string
        ostringstream oss;
        oss << "Distance: " << center_distance << " cm";
        string distance_text = oss.str();

        // Draw distance text on image
        putText(left, distance_text, center,
                FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 2, 8);

        //--------------------- DISPLAY WINDOWS ---------------------
        // Show all outputs
        imshow("left", left);
        imshow("right", right);
        imshow("disparity", disp8);
        imshow("distance_map", distance_map);

        // Wait briefly and allow exit with ESC key
        if (waitKey(10) == 27) {
            break;
        }
    }

    return 0;
}
