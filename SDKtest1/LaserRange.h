#pragma once
#include "iostream"
#include "stdio.h"
#include "stdint.h"
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"
#include "cxcore.h"

using namespace std;



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
	int bright;
	int Pixnum;

} MPoint;

