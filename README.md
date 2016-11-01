#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std; 

Point centerPoint; 
Mat weight;
//Mat context;



void getContextPrior(Mat& _frame,Rect& _rect)
{
	double dist;
	double sigma=(_rect.width+_rect.height)*0.5;

	centerPoint.x=_rect.x+_rect.width*0.5;
	centerPoint.y=_rect.y+_rect.height*0.5;

	weight=Mat::zeros(_frame.rows,_frame.cols,CV_8UC1); 

	for (int r=0;r<_frame.rows;r++)
	{
		for (int c=0;c<_frame.cols;c++)
		{
			double x=c;
			double y=r; 
			dist=sqrt((y-centerPoint.y)*(y-centerPoint.y)+(x-centerPoint.x)*(x-centerPoint.x));
			weight.at<char>(r,c)=cvRound(exp(-dist*dist/(2*sigma*sigma))); 
		}
	}

	//context=weight.mul(_frame);
}



Mat image;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;

static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}

static void help()
{
    cout << "\nThis is a demo that shows mean-shift based tracking\n"
            "You select a color objects such as your face and it tracks it.\n"
            "This reads from video camera (0 by default, or the camera number the user enters\n"
            "Usage: \n"
            "   ./camshiftdemo [camera number]\n";

    cout << "\n\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tc - stop the tracking\n"
            "\tb - switch to/from backprojection view\n"
            "\th - show/hide object histogram\n"
            "\tp - pause video\n"
            "To initialize tracking, select the object with mouse\n";
}

const char* keys =
{
    "{1|  | 0 | camera number}"
};

int main( int argc, const char** argv )
{
    help();

    VideoCapture cap;
    Rect trackWindow,trackWindowg;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    CommandLineParser parser(argc, argv, keys);
    int camNum = parser.get<int>("1");

    cap.open(camNum);

    if( !cap.isOpened() )
    {
        help();
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        parser.printParams();
        return -1;
    }

	/*
	// 测试摄像头可用否
	while (1) {   
		Mat c; //定义一个Mat变量，用于存储每一帧的图像  
		cap>>c; //读取当前帧  
		if (!c.empty()) //判断当前帧是否捕捉成功 **这步很重要  
			imshow("test", c); //若当前帧捕捉成功，显示  
		else  
			cout<<"can not ";   
		waitKey(30); //延时30毫秒  
	} 
	*/
    //namedWindow( "Histogram", 0 );
    namedWindow( "CamShift Demo", 0 );
    setMouseCallback( "CamShift Demo", onMouse, 0 );
    //createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
    //createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
    //createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );

    Mat frame, hsv,gray, hue, mask, maskg,hist, histg, backproj,backprojg;
    bool paused = false;

    while (1)
    {
        if( !paused )
        {
            cap >> frame;
            if( !frame.empty())
               imshow( "CamShift Demo", frame);
			else break;
        }

		
        frame.copyTo(image);

        if( !paused )
        {
            cvtColor(image, hsv, CV_BGR2HSV); 

			cvtColor(image, gray, CV_BGR2GRAY); 

            if( trackObject )
            {
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);   //int vmin = 10, vmax = 256, smin = 30;
		
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if( trackObject < 0 )
                {
                    Mat roi(hue, selection), maskroi(mask, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, CV_MINMAX);  //线性归一化

                    trackWindow = selection;
					trackWindowg = selection;
                    trackObject = 1;
					getContextPrior(gray,selection);
                }

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
				backproj &=weight;
				//backproj &= context;
             


              CamShift(backproj, trackWindow,TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
                if( trackWindow.area() <= 1 )
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }
				getContextPrior(gray,trackWindow);
             
                if( backprojMode )
                    cvtColor( backproj, image, CV_GRAY2BGR );
				rectangle(image,trackWindow,Scalar(0,0,255),6);
            }

			
        }
        else if( trackObject < 0 )
            paused = false;

        if( selectObject && selection.width > 0 && selection.height > 0 )
        {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }

        // imshow( "CamShift Demo", image );
//        imshow( "Histogram", histimg );

        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch(c)
        {
        case 'b':
            backprojMode = !backprojMode;
            break;
        case 'c':
            trackObject = 0;
//            histimg = Scalar::all(0);
            break;
        case 'h':
            showHist = !showHist;
            if( !showHist )
                destroyWindow( "Histogram" );
            else
                namedWindow( "Histogram", 1 );
            break;
        case 'p':
            paused = !paused;
            break;
        default:
            ;
        }
		
    }
	waitKey(0);
    return 0;
}

