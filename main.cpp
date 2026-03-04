task 1 final code with comments


Summarise

(s) Jamie Gatehouse

​
(s) Jamie Gatehouse​
// Standard input/output stream library
#include <iostream>

// File stream library (not used in this code but included)
#include <fstream>

// System data types
#include <sys/types.h>

// String handling
#include <string>

// Math functions (sqrt, pow, etc.)
#include <cmath>

// Custom owl robot library
#include "../owl.h"

using namespace std;
using namespace cv;   // OpenCV namespace

int main()
{
    // Create robotOwl object
    // Parameters likely represent servo calibration values
    // Final "true" probably enables camera or robot hardware
    robotOwl owl(1500, 1475, 1520, 1525, 1520, true);

    // Infinite loop to continuously capture and process frames
    while (true)
    {
        // Create matrices to hold left and right camera frames
        Mat left, right;

        // Capture frames from the owl's stereo cameras
        owl.getCameraFrames(left, right);

        // Calculate centre point of left image
        Point centrePoint(left.cols / 2, left.rows / 2);

        // ===============================
        // ===== 3x3 PIXEL AVERAGING =====
        // ===============================
        // Instead of sampling a single pixel,
        // we average a 3x3 square to reduce noise.

        int redSum = 0;
        int greenSum = 0;
        int blueSum = 0;

        // Loop through 3x3 area around centre pixel
        for(int y = -1; y <= 1; y++)
        {
            for(int x = -1; x <= 1; x++)
            {
                // Get neighbouring pixel position
                Point p = centrePoint + Point(x, y);

                // Get BGR pixel value at that location
                Vec3b pixelValue = left.at<Vec3b>(p);

                // OpenCV stores colour as BGR (not RGB)
                blueSum  += pixelValue[0];
                greenSum += pixelValue[1];
                redSum   += pixelValue[2];
            }
        }

        // Compute average RGB values
        unsigned char red   = redSum / 9;
        unsigned char green = greenSum / 9;
        unsigned char blue  = blueSum / 9;

        // ===============================
        // ===== DRAWING OVERLAY =====
        // ===============================

        // Draw white circle at centre point
        circle(left, centrePoint, 15, Scalar(255,255,255), 2);

        // Create string displaying RGB values
        string rgbText = "(" + to_string(red) + ", " +
                               to_string(green) + ", " +
                               to_string(blue) + ")";

        // Display RGB text below centre
        putText(left, rgbText,
                centrePoint + Point(-100,50),
                FONT_HERSHEY_SIMPLEX, 1,
                Scalar(255,255,255), 2);

        // ==================================
        // ===== RATIO-BASED DETECTION =====
        // ==================================
        // Convert raw RGB values to ratios.
        // This reduces lighting sensitivity.

        float r = red;
        float g = green;
        float b = blue;

        float sum = r + g + b;

        // Prevent division by zero
        if(sum == 0) sum = 1;

        // Calculate normalised colour ratios
        float rRatio = r / sum;
        float gRatio = g / sum;
        float bRatio = b / sum;

        // Store current colour as vector
        Vec3f current(rRatio, gRatio, bRatio);

        // Reference colour ratios
        Vec3f whiteRef(0.33f, 0.33f, 0.33f);
        Vec3f redRef  (1.0f,  0.0f,  0.0f);
        Vec3f greenRef(0.0f,  1.0f,  0.0f);
        Vec3f blueRef (0.0f,  0.0f,  1.0f);

        // Lambda function to compute Euclidean distance
        // between two colour ratio vectors
        auto colorDistance = [](Vec3f a, Vec3f b)
        {
            return sqrt(pow(a[0]-b[0],2) +
                        pow(a[1]-b[1],2) +
                        pow(a[2]-b[2],2));
        };

        string detectedColor = "";
        Scalar textColor;     // Colour for drawing text
        float confidence = 0.0f;

        // Detect black separately using brightness threshold
        if(red + green + blue < 60)
        {
            detectedColor = "Black";
            textColor = Scalar(0,0,0);
            confidence = 100.0f;  // Very dark = strong black confidence
        }
        else
        {
            // Calculate distance from each reference colour
            float dWhite = colorDistance(current, whiteRef);
            float dRed   = colorDistance(current, redRef);
            float dGreen = colorDistance(current, greenRef);
            float dBlue  = colorDistance(current, blueRef);

            // Find smallest distance
            float minDist = min(min(dWhite, dRed), min(dGreen, dBlue));

            // Convert distance to confidence percentage
            // Max distance in ratio space ≈ sqrt(2)
            float maxDist = sqrt(2.0f);
            confidence = (1.0f - (minDist / maxDist)) * 100.0f;

            // Determine which colour was closest
            if(minDist == dWhite)
            {
                detectedColor = "White";
                textColor = Scalar(255,255,255);
            }
            else if(minDist == dRed)
            {
                detectedColor = "Red";
                textColor = Scalar(0,0,255);   // BGR format
            }
            else if(minDist == dGreen)
            {
                detectedColor = "Green";
                textColor = Scalar(0,255,0);
            }
            else
            {
                detectedColor = "Blue";
                textColor = Scalar(255,0,0);
            }
        }

        // Display detected colour above centre
        putText(left, detectedColor,
                centrePoint + Point(-75,-50),
                FONT_HERSHEY_SIMPLEX, 2,
                textColor, 5);

        // ===============================
        // ===== DISPLAY CONFIDENCE =====
        // ===============================

        string confText = "Confidence: " + to_string((int)confidence) + "%";

        putText(left, confText,
                centrePoint + Point(-100,90),
                FONT_HERSHEY_SIMPLEX, 0.9,
                Scalar(255,255,0), 2);

        // Show processed left camera image
        imshow("left", left);

        // Wait 30ms before next frame (approx 33 FPS)
        waitKey(30);
    }
}
