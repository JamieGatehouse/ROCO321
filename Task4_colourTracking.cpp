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
    robotOwl owl(1500, 1475, 1520, 1525, 1520);

    //--------------------- LOAD CALIBRATION FILES ---------------------
    // File paths for intrinsic and extrinsic calibration
    string intrinsic_filename = "../intrinsics.xml";
    string extrinsic_filename = "../extrinsics.xml";

    // Rectangles for valid image regions after rectification
    Rect roi1, roi2;

    // Q matrix used for depth reconstruction
    Mat Q;

    // Image size (must match calibration settings)
    Size img_size = {640, 480};

    // Open intrinsic calibration file
    FileStorage fs(intrinsic_filename, FileStorage::READ);
    if (!fs.isOpened()) {
        printf("Failed to open file %s\n", intrinsic_filename.c_str());
        return -1;
    }

    // Camera matrices and distortion coefficients
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

    // Rotation and translation between cameras
    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    //--------------------- STEREO RECTIFICATION ---------------------
    // Align stereo images so corresponding points lie on same row
    stereoRectify(M1, D1, M2, D2, img_size, R, T,
                  R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

    // Precompute maps for undistortion and rectification
    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    //--------------------- DISTANCE CALIBRATION ---------------------
    // Known distances (cm) and corresponding disparities (pixels)
    vector<float> known_distances = {50.0, 100.0, 150.0};
    vector<float> known_disparities = {100.0, 50.0, 25.0};

    // Constant used for distance calculation
    float C = 0;

    // Compute calibration constant C = disparity * distance
    for (size_t i = 0; i < known_distances.size(); ++i) {
        C = known_disparities[i] * known_distances[i];
    }

    //--------------------- CREATE STEREO MATCHER ---------------------
    // Parameters for StereoSGBM algorithm
    int SADWindowSize = 5;
    int numberOfDisparities = 256;

    // Create StereoSGBM object
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

    // Configure matching parameters
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
        // Matrices for stereo images
        Mat left, right;

        // Capture frames from cameras
        owl.getCameraFrames(left, right);

        // Check for capture failure
        if (left.empty() || right.empty()) {
            cerr << "Failed to capture frames!" << endl;
            break;
        }

        //--------------------- IMAGE RECTIFICATION ---------------------
        // Correct distortion and align images
        remap(left, left, map11, map12, INTER_LINEAR);
        remap(right, right, map21, map22, INTER_LINEAR);

        //--------------------- DISPARITY COMPUTATION ---------------------
        // Compute disparity map
        Mat disp, disp8;
        sgbm->compute(left, right, disp);

        // Convert to 8-bit for display only
        disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * 16.0));

        //--------------------- DISTANCE MAP CREATION ---------------------
        // Create floating-point distance map
        Mat distance_map = Mat::zeros(disp.size(), CV_32F);

        // Convert disparity values to distance
        for (int y = 0; y < disp.rows; ++y) {
            for (int x = 0; x < disp.cols; ++x) {

                // Ignore invalid disparity values
                if (disp.at<short>(y, x) > 0) {

                    // Convert disparity from fixed-point format
                    float disparity = disp.at<short>(y, x) / 16.0f;

                    // Calculate distance using calibration constant
                    float distance = C / disparity;

                    // Store result in distance map
                    distance_map.at<float>(y, x) = distance;
                }
            }
        }

        //--------------------- YELLOW OBJECT DETECTION ---------------------
        // Convert image to HSV colour space
        Mat hsv, mask;
        cvtColor(left, hsv, COLOR_BGR2HSV);

        // Define HSV range for yellow colour
        Scalar lower_yellow(20, 100, 100);
        Scalar upper_yellow(30, 255, 255);

        // Create binary mask for yellow regions
        inRange(hsv, lower_yellow, upper_yellow, mask);

        //--------------------- MORPHOLOGICAL CLEANUP ---------------------
        // Remove noise using erosion and dilation
        erode(mask, mask, Mat(), Point(-1,-1), 2);
        dilate(mask, mask, Mat(), Point(-1,-1), 2);

        //--------------------- CONTOUR DETECTION ---------------------
        // Find contours in the binary mask
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Rect boundingBox;

        // If any contours found, select the largest one
        if (!contours.empty()) {
            int maxIdx = 0;
            double maxArea = contourArea(contours[0]);

            for (size_t i = 1; i < contours.size(); ++i) {
                double area = contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxIdx = i;
                }
            }

            // Create bounding box around largest contour
            boundingBox = boundingRect(contours[maxIdx]);

            // Draw bounding box on image
            rectangle(left, boundingBox, Scalar(0, 255, 0), 2);
        }

        //--------------------- DISTANCE AT TARGET ---------------------
        // Determine point to measure distance
        Point center;

        // Use bounding box centre if object detected
        if (boundingBox.area() > 0) {
            center = Point(boundingBox.x + boundingBox.width/2,
                           boundingBox.y + boundingBox.height/2);
        }
        else {
            // Fallback to image centre
            center = Point(left.cols/2, left.rows/2);
        }

        // Retrieve distance value at chosen point
        float center_distance = distance_map.at<float>(center);

        //--------------------- DISPLAY DISTANCE ---------------------
        // Convert distance to string
        ostringstream oss;
        oss << "Distance: " << center_distance << " cm";

        // Display distance on image
        putText(left, oss.str(), center,
                FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 2);

        //--------------------- DISPLAY WINDOWS ---------------------
        // Show all outputs
        imshow("left", left);
        imshow("right", right);
        imshow("disparity", disp8);
        imshow("distance_map", distance_map);
        imshow("yellow_mask", mask); // debug view

        // Exit on ESC key
        if (waitKey(10) == 27) break;
    }

    return 0;
}
