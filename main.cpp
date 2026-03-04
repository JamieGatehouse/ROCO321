task 2: code with comments


Summarise

(s) Jamie Gatehouse

​
(s) Jamie Gatehouse​
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <string>
#include <deque>

#include "../owl.h"

using namespace std;
using namespace cv;

int main()
{
    // Initialise the robotOwl object with starting servo calibration values
    // Format: robotOwl(servo1, servo2, servo3, servo4, servo5)
    robotOwl owl(1525, 1560, 1540, 1540, 1465);

    // Store previous tracked positions for drawing a motion trail
    deque<Point> trail;

    // Maximum number of previous positions to keep
    const int maxTrailSize = 20;

    // =========================
    // SERVO MOVEMENT LIMITS
    // =========================

    // Minimum and maximum pan (left-right) values
    const int minPan  = 1180;
    const int maxPan  = 1850;

    // Minimum and maximum tilt (up-down) values
    const int minTilt = 1180;
    const int maxTilt = 2000;

    // Keep track of the current absolute servo positions
    int currentPan  = 1525;
    int currentTilt = 1560;

    // Direction of horizontal scanning
    // 1 = moving right, -1 = moving left
    int panDirection = 1;

    // Step sizes for scanning movement
    const int panStep  = 60;   // horizontal scan speed
    const int tiltStep = 60;   // vertical scan step size

    while (true)
    {
        // Capture frames from the owl's stereo cameras
        Mat left, right;
        owl.getCameraFrames(left, right);

        // Convert left image to HSV colour space
        Mat leftHSV, leftMask;
        cvtColor(left, leftHSV, COLOR_BGR2HSV);

        // Threshold HSV image to isolate target colour range
        // (Currently set for yellow objects)
        inRange(leftHSV, Vec3b(25,127,127), Vec3b(35,255,255), leftMask);

        // Calculate image moments to determine object centre
        Moments m = moments(leftMask, true);

        // =========================
        // IF TARGET IS DETECTED
        // =========================
        if (m.m00 != 0.0)
        {
            // Compute centre of detected object
            Point centre(m.m10/m.m00, m.m01/m.m00);

            // Add centre to trail history
            trail.push_back(centre);

            // Ensure trail does not exceed max size
            if (trail.size() > maxTrailSize)
                trail.pop_front();

            // Draw circle at detected object centre
            circle(left, centre, 10, Scalar(0,0,255), 2);

            // Draw trail connecting previous positions
            for (size_t i = 1; i < trail.size(); i++)
                line(left, trail[i-1], trail[i], Scalar(255,0,0), 2);

            // Get centre of camera frame
            int frameCentreX = left.cols / 2;
            int frameCentreY = left.rows / 2;

            // Calculate positional error from frame centre
            int errorX = centre.x - frameCentreX;
            int errorY = centre.y - frameCentreY;

            // Proportional gain for servo movement
            float gain = 0.2;

            int moveX = 0;
            int moveY = 0;

            // Only move if error is significant (deadzone = 10 pixels)
            if (abs(errorX) > 10)
                moveX = errorX * gain;

            if (abs(errorY) > 10)
                moveY = -errorY * gain;

            // Update stored absolute servo positions
            currentPan  += moveX;
            currentTilt += moveY;

            // Clamp positions so servos never exceed safe limits
            currentPan  = max(minPan,  min(maxPan,  currentPan));
            currentTilt = max(minTilt, min(maxTilt, currentTilt));

            // Move servos relative to current position
            owl.setServoRelativePositions(0, 0, moveX, moveY, 0);
        }
        else
        {
            // =========================
            // IF NO TARGET IS DETECTED
            // ENTER FULL ENVIRONMENT SCAN MODE
            // =========================

            // Clear trail since object is lost
            trail.clear();

            // Continue sweeping horizontally
            int moveX = panStep * panDirection;
            int moveY = 0;

            currentPan += moveX;

            // If horizontal limit reached
            if (currentPan >= maxPan || currentPan <= minPan)
            {
                // Reverse horizontal direction
                panDirection *= -1;

                moveX = panStep * panDirection;

                // Step down vertically after each horizontal sweep
                moveY = tiltStep;
                currentTilt += tiltStep;

                // If bottom limit reached, restart from top
                if (currentTilt > maxTilt)
                {
                    moveY = minTilt - currentTilt;
                    currentTilt = minTilt;
                }
            }

            // Apply scanning movement
            owl.setServoRelativePositions(0, 0, moveX, moveY, 0);
        }

        // Display live output windows
        imshow("left", left);
        imshow("HSV", leftHSV);
        imshow("Mask", leftMask);

        // Small delay to allow window refresh and servo stability
        waitKey(10);
    }

    return 0;
}
