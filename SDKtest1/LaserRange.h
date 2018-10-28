#pragma once
#include "iostream"
#include "stdio.h"
#include "stdint.h"
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"
#include "cxcore.h"
#include <opencv2/opencv.hpp>
#include <process.h>
#include "afxwin.h"
#include "math.h"
#include "cstdlib"

#include "ImProcess.h"


using namespace std;
using namespace cv;


class LaserRange
{
public:
	typedef struct RangeResult {
		unsigned int maxCol;
		unsigned int maxRow;
		unsigned int maxPixel;
		double Range;
		unsigned int  PixfromCent;
	} RangeResult;
	RangeResult *GetRange(IplImage *imgRange, IplImage *imgDst);
	LaserRange();
	virtual ~LaserRange();

private:
	unsigned int maxW;
	unsigned int maxH;
	unsigned int MaxPixel;
	RangeResult *strctResult;

	//value used for calculating range from captured image data
	const double gain;//gain constant used for converting pixel offset to angle in radians
	const double offset;//offset constant
	const double h_cm;//distance between center of camera and laser
	unsigned int pixel_from_center;//brightest pixel location from center

	void Preprocess(void *img, IplImage *imgTemp);
};

class CLaserVisionDlg
{
public:
	int CaptureImage(IplImage *iplimage);
};

typedef struct MPoint
{
	int x;
	int y;
	double cx;
	double cy;
	int bright;
	int Pixnum;

} MPoint;

//边缘测试算法1
void getPeaker1(Mat matImage, MPoint *point);
//Canny opencv 边缘检测
void getcanny(Mat matImage, MPoint *point);
//亚像素分析函数（待完善）
void getdoublepixel(Mat matImage, MPoint *point);
//基于double的误差标记函数
void getErrorIdentifyDouble(Mat matImage, MPoint *point);

//双精度平均值计算
double average(int *x, int len);
//误差标记函数
void getErrorIdentifyInt(Mat matImage, MPoint *point);
//被giveup的寻址method1
void getDiff1test(Mat matImage, MPoint *point);
//张正友相机标定法
void calibfirst(Mat matImage);

