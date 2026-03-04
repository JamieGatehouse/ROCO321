#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <iostream>
#include <string>

#include "../owl.h"

using namespace std;
using namespace cv;

int main()
{
    //connect with the owl and load calibration values
    robotOwl owl(1525, 1560, 1540, 1540, 1465, true); //starts in "quiet mode" which switches off the servos.

    while (true){
        //read the owls camera frames
        Mat left, right;
        owl.getCameraFrames(left, right);

        //get pixel colour values
        Point centrePoint(left.size().width/2, left.size().height/2);
        Vec3b pixelValue = left.at<Vec3b>(centrePoint);
        unsigned char red  =pixelValue[2];
        unsigned char green=pixelValue[1];
        unsigned char blue =pixelValue[0];

        //drawing functions
        circle(left, centrePoint, 15, Scalar(255,255,255), 2); //draw a circle to show the pixel being measured
        string text = "(" + to_string(red) + ", " + to_string(green) + ", " + to_string(blue) + ")";      //create a string of the RGB values
        putText(left, text, centrePoint+Point(-100,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2); //draw the string to the image


        //Detect Black when each RGB value is below 50
        if (red<50 && green<50 && blue<50)
        putText(left, "Black", centrePoint+Point(-75, -50), FONT_HERSHEY_SIMPLEX, 2, Scalar (0,255, 255), 5);

        //Detect White when each RGB value is above 205
        if (red> 205 && green>205 && blue>205)
        putText(left, "White", centrePoint+Point(-75, -50), FONT_HERSHEY_SIMPLEX, 2, Scalar (0,255, 255), 5);

        //Detect Red when the R value is hight and the GB values are low
        if(red>205 && green<100 && blue<100)
        putText(left, "Red", centrePoint+Point(-75, -50), FONT_HERSHEY_SIMPLEX, 2, Scalar (0,255, 255), 5) ;

        //Detect Green when the G value is high and the RB values are low
        if(red<100 && green>205 && blue<100)
        putText(left, "Green", centrePoint+Point(-75,-50), FONT_HERSHEY_SIMPLEX, 2, Scalar (0,255, 255), 5) ;

        //Detect Blue when the B value is high and the RG values are low
        if (red<50 && green<120 && blue>205)
        putText(left, "Blue", centrePoint+Point(-75, -50), FONT_HERSHEY_SIMPLEX, 2, Scalar (0,255, 255) , 5);


        //display image
        imshow("left",left);
        waitKey(30);
    }
}
