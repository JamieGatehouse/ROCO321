#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <string>
#include <deque>
#include <sstream>
#include <cmath>

#include "../owl.h"

using namespace std;
using namespace cv;

#define targetSize 60 //adjust this to change the size of your target

int main()
{
    //--------------------- INITIALISE ROBOT ---------------------
    // Create robotOwl object with calibration values for servos
    robotOwl owl(1500, 1475, 1520, 1525, 1520);

    //--------------------- TARGET SELECTION LOOP ---------------------
    // Mat to store selected target image
    Mat target;

    // Define a fixed square in the centre of the image
    Rect targetPos(320-targetSize/2, 240-targetSize/2, targetSize, targetSize);

    while (true){
        // Matrices to store stereo camera frames
        Mat left, right;

        // Capture frames from both cameras
        owl.getCameraFrames(left, right);

        // If spacebar is pressed, capture the target region
        if(waitKey(10)==' ')
        {
            // Copy selected region into target image
            left(targetPos).copyTo(target);

            cout<<"Target selected!"<<endl;
            break;
        }

        // Draw selection rectangle on screen
        rectangle(left, targetPos, Scalar(255,255,255), 2);

        // Display left camera feed
        imshow("left",left);
    }

    //--------------------- COMPUTE INITIAL TARGET COLOUR ---------------------
    // Compute average colour of selected target region
    Scalar meanColor = mean(target);

    // Extract BGR values (OpenCV uses BGR format)
    int avgB = (int)meanColor[0];
    int avgG = (int)meanColor[1];
    int avgR = (int)meanColor[2];

    // Print initial target colour
    cout << "Initial target average RGB: (" << avgR << ", " << avgG << ", " << avgB << ")" << endl;

    //--------------------- TRACKING LOOP ---------------------
    cout<<"Starting tracking code..."<<endl;

    // Template matching threshold (lower = stricter match)
    const double threshold = 0.3;

    // Gain controls how aggressively servos move
    const float gain = 0.05f;

    // Distance between cameras (used for triangulation)
    const float baseline = 0.067f;

    // Maximum allowed RGB difference for colour validation
    const int rgbTolerance = 30;

    // Store previous positions to draw motion trail
    deque<Point> trail;

    // Maximum trail length
    const int maxTrailSize = 20;

    while(1)
    {
        // Capture new frames
        Mat left, right;
        owl.getCameraFrames(left, right);

        //--------------------- CONVERT TO GRAYSCALE ---------------------
        // Convert images to grayscale for template matching
        Mat leftGray, rightGray, targetGray;
        cvtColor(left, leftGray, COLOR_BGR2GRAY);
        cvtColor(right, rightGray, COLOR_BGR2GRAY);
        cvtColor(target, targetGray, COLOR_BGR2GRAY);

        //--------------------- TEMPLATE MATCHING (LEFT) ---------------------
        // Perform template matching on left image
        Mat resultL;
        matchTemplate(leftGray, targetGray, resultL, TM_SQDIFF_NORMED);

        double minValL, maxValL;
        Point minLocL, maxLocL;

        // Find best match location (minimum value)
        minMaxLoc(resultL, &minValL, &maxValL, &minLocL, &maxLocL);

        //--------------------- TEMPLATE MATCHING (RIGHT) ---------------------
        // Perform template matching on right image
        Mat resultR;
        matchTemplate(rightGray, targetGray, resultR, TM_SQDIFF_NORMED);

        double minValR, maxValR;
        Point minLocR, maxLocR;

        minMaxLoc(resultR, &minValR, &maxValR, &minLocR, &maxLocR);

        //--------------------- CHECK MATCH QUALITY ---------------------
        // Only proceed if both matches are good enough
        if(minValL < threshold && minValR < threshold)
        {
            //--------------------- COLOUR VALIDATION ---------------------
            // Extract matched region from left image
            Rect roi(minLocL.x, minLocL.y, target.cols, target.rows);
            Mat trackedRegion = left(roi);

            // Compute current average colour
            Scalar currentMean = mean(trackedRegion);

            int curB = (int)currentMean[0];
            int curG = (int)currentMean[1];
            int curR = (int)currentMean[2];

            // Compare with original target colour
            bool colorMismatch = (abs(curR - avgR) > rgbTolerance) ||
                                 (abs(curG - avgG) > rgbTolerance) ||
                                 (abs(curB - avgB) > rgbTolerance);

            // Choose rectangle colour based on match quality
            Scalar rectColor = colorMismatch ? Scalar(0,0,255) : Scalar(0,255,0);

            // Draw bounding boxes on both images
            rectangle(left, minLocL,
                      Point(minLocL.x + target.cols, minLocL.y + target.rows),
                      rectColor, 2);

            rectangle(right, minLocR,
                      Point(minLocR.x + target.cols, minLocR.y + target.rows),
                      rectColor, 2);

            //--------------------- COMPUTE CENTRES ---------------------
            // Compute centre of detected target in both images
            int centreXL = minLocL.x + target.cols/2;
            int centreXR = minLocR.x + target.cols/2;
            int centreYL = minLocL.y + target.rows/2;

            // Compute centre of image
            int imageCentreX = left.cols/2;

            //--------------------- ERROR CALCULATION ---------------------
            // Error is difference between image centre and target centre
            int errorL = imageCentreX - centreXL;
            int errorR = imageCentreX - centreXR;

            //--------------------- SERVO CONTROL ---------------------
            // Adjust servo positions to keep target centred
            owl.setServoRelativePositions(
                -gain * errorR, 0,
                -gain * errorL, 0,
                0
            );

            //--------------------- TRAIL DRAWING ---------------------
            // Store current position
            Point currentPos(centreXL, centreYL);
            trail.push_back(currentPos);

            // Limit trail size
            if(trail.size() > maxTrailSize)
                trail.pop_front();

            // Draw motion trail (yellow lines)
            for(size_t i = 1; i < trail.size(); i++)
            {
                line(left, trail[i-1], trail[i], Scalar(0,255,255), 2);
            }

            //--------------------- DISTANCE ESTIMATION ---------------------
            // Get current servo angles
            float angleL, angleR;
            owl.getServoAngles(angleL, angleR);

            // Compute denominator for triangulation
            float denom = tan(angleL) + tan(angleR);

            int textY = 30;

            // Avoid division by zero
            if(fabs(denom) > 0.0001)
            {
                // Calculate distance using stereo geometry
                float distance = baseline / fabs(denom);

                // Display distance on image
                stringstream ss;
                ss << "Distance: " << distance << " m";

                putText(left, ss.str(), Point(30, textY),
                        FONT_HERSHEY_SIMPLEX, 0.7,
                        Scalar(0,255,0), 2);

                cout << "Estimated distance: " << distance << " m" << endl;

                textY += 30;
            }
            else
            {
                cout << "Invalid angle combination (division avoided)" << endl;
            }

            //--------------------- DISPLAY TARGET COLOUR ---------------------
            // Show original target colour for reference
            stringstream ssColor;
            ssColor << "Target Avg RGB: (" << avgR << ", " << avgG << ", " << avgB << ")";

            putText(left, ssColor.str(), Point(30, textY),
                    FONT_HERSHEY_SIMPLEX, 0.7,
                    Scalar(255,255,255), 2);
        }
        else
        {
            // If target is lost, clear motion trail
            trail.clear();
            cout << "Target not found" << endl;
        }

        //--------------------- DISPLAY WINDOWS ---------------------
        // Show all relevant windows
        imshow("left", left);
        imshow("right", right);
        imshow("target", target);

        // Small delay for frame update
        waitKey(10);
    }
}
