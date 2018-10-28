// SDKtest1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "stdafx.h"
//#include "windows.h"
#include <opencv2/opencv.hpp>
#include "cv.h"
#include <process.h>
#include "CameraApi.h"
#include "LaserRange.h"
#include "afxwin.h"
#include "math.h"
#include "cstdlib"
#include "sstream"
#include "ImProcess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#ifdef _WIN64
#pragma comment(lib, "MVCAMSDK_X64.lib")
#else
#pragma comment(lib, "..\\MVCAMSDK.lib")
#endif
//#include "..//include//CameraApi.h"

using namespace std;
using namespace cv;

UINT            m_threadID;		//图像抓取线程的ID
HANDLE          m_hDispThread;	//图像抓取线程的句柄
BOOL            m_bExit = FALSE;//用来通知图像抓取线程结束
CameraHandle    m_hCamera;		//相机句柄，多个相机同时使用时，可以用数组代替	
BYTE*           m_pFrameBuffer; //用于将原始图像数据转换为RGB的缓冲区
tSdkFrameHead   m_sFrInfo;		//用于保存当前图像帧的帧头信息

int	            m_iDispFrameNum;	//用于记录当前已经显示的图像帧的数量
float           m_fDispFps;			//显示帧率
float           m_fCapFps;			//捕获帧率
tSdkFrameStatistic  m_sFrameCount;
tSdkFrameStatistic  m_sFrameLast;
int					m_iTimeLast;
char		    g_CameraName[64];
//#define USE_CALLBACK_GRAB_IMAGE
#define CALIB_CAMERA
//#define GET_CANNY
//#define GET_DOUBLE_PIXEL

/*
USE_CALLBACK_GRAB_IMAGE
如果需要使用回调函数的方式获得图像数据，则反注释宏定义USE_CALLBACK_GRAB_IMAGE.
我们的SDK同时支持回调函数和主动调用接口抓取图像的方式。两种方式都采用了"零拷贝"机制，以最大的程度的降低系统负荷，提高程序执行效率。
但是主动抓取方式比回调函数的方式更加灵活，可以设置超时等待时间等，我们建议您使用 uiDisplayThread 中的方式
*/
//#define USE_CALLBACK_GRAB_IMAGE 

void getPeaker1(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	int MaxPixel;

	/*method 1: the average of the xPisel with same brightness*/
	int Pixeldata;
	int x1 = 0;
	int x2 = 0;
	float step = 0.85;//边沿值百分数
	boolean stepon(false), stepoff(false);
	int getpeak = 0;

	for (int j = 0; j < Rows; j++) {
		uchar* data = matImage.ptr<uchar>(j);
		MaxPixel = data[0];
		for (int i = 1; i < Cols; i++) {
			//data[i] = data[i]; /// div * div + div / 2;
			if (data[i] > MaxPixel) {
				MaxPixel = data[i];
				//point[j].x = i;
			}
		}
		//point[j].y = j;
		point[j].bright = MaxPixel;
		for (int i = 0; i < Cols; i++) {
			Pixeldata = data[i];
			if (stepon == false && Pixeldata > step*MaxPixel) {
				x1 = i;
				stepon = true;
			}
			else if (stepon = true && Pixeldata == MaxPixel && stepoff == false) {
				getpeak++;
			}
			else if (getpeak && Pixeldata < step*MaxPixel) {
				x2 = i - 1;
				stepoff == true;
				break;
			}
		}
		point[j].Pixnum = getpeak;
		point[j].x = (x1 + x2) / 2;
		point[j].y = j;
		stepon = false;
		stepoff = false;
		getpeak = 0;
		x1 = 0;
		x2 = 0;
		cout << "(" << point[j].x << "," << point[j].y << "):" << MaxPixel << "    sum:" << point[j].Pixnum << endl;
	}
}

//Canny opencv 边缘检测
void getcanny(Mat matImage, MPoint *point) {
	int g_nCannyLowThreshold = 100;//canny检测低阈值
	Mat tmpImage,dstImage;
	blur(matImage, tmpImage, Size(3, 3));
	Canny(tmpImage,dstImage, g_nCannyLowThreshold, g_nCannyLowThreshold*3);
	namedWindow("canny function");
	imshow("canny function",dstImage);
	int Rows = matImage.rows;
	int Cols = matImage.cols*matImage.channels();
	int x[100];
	int px = 0 ;
	int PixelDataof;
	int sum = 0;
	double average = 0;
	for (int j = 0; j < Rows; j++) {
		uchar* data = dstImage.ptr<uchar>(j);
		for (int i = 0; i < Cols; i++) {
			PixelDataof = data[i];
			if (PixelDataof>130) {
				x[px] = i;
				px++;
				sum = sum + i;
				if (px > 100) {
					cout << "there are too many canny points" << endl;
				}
			}
		}
		//逐行计算平均点
		if (px) {
			average = sum*1.0 / px;
		}
		point[j].cx = average;
		point[j].cy = j;
		average = 0;
		sum = 0;
		px = 0;
		memset(x, 0, px);
		//cout << "(" << point[j].cx << "," << point[j].cy << ")" << endl;
	}

}
//亚像素分析函数（待完善）
void getdoublepixel(Mat matImage, MPoint *point) {
	IplImage* image = NULL;
	IplImage* pGrayImage = NULL;
	IplImage* pEdgeImage = NULL;
	CvMat* pDirection = NULL;
	CvMat* pSubEdgeMatH = NULL;
	CvMat* pSubEdgeMatW = NULL;
	CvMat* pGradMat = NULL;
	int nWidth = 0, nHeight = 0;
	CHTimer timer;
	//cvNamedWindow("Orign Image", 1);
	cvNamedWindow("Image", 1);
	image = &IplImage(matImage);
	//cvShowImage("Orign Image", image);
	//Mat OrImage = cvarrToMat(image);
	pGrayImage = cvCreateImage(cvSize(image->width, image->height), image->depth, 1);
	pEdgeImage = cvCreateImage(cvSize(image->width, image->height), image->depth, 1);
	pDirection = cvCreateMat(image->height, image->width, CV_8SC1);
	pSubEdgeMatH = cvCreateMat(image->height, image->width, CV_64FC1);
	pSubEdgeMatW = cvCreateMat(image->height, image->width, CV_64FC1);
	pGradMat = cvCreateMat(image->height, image->width, CV_32SC2);
#if 0
	cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
	cout << "The channels of the image:" << image->nChannels << endl;
#endif
	nWidth = image->width;
	nHeight = image->height;
	if (image->nChannels == 3)
	{
		cvCvtColor(image, pGrayImage, CV_RGB2GRAY);
		cvSmooth(pGrayImage, pGrayImage, CV_MEDIAN, 3, 0, 0, 0);
	}
	else if (image->nChannels == 1)
	{
		cvSmooth(image, pGrayImage, CV_MEDIAN, 3, 0, 0, 0);
	}
	//timer.StartTime();
	//canny边缘算法
	cvCanny(pGrayImage, pEdgeImage, 120, 180);
	//sobel边缘算法
	hSobel(pGrayImage, pEdgeImage, pGradMat, pDirection, 10, 120);
	//亚像素分析
	hSubPixelEdge(pGradMat, pEdgeImage, pDirection, pSubEdgeMatH, pSubEdgeMatW);
	//timer.EndTime();
	//cout << "time:" << timer.GetTime() << "ms" << endl;
	cvShowImage("Image", pEdgeImage);

	////对生成的图像进行解析	
    int Rows = matImage.rows;//y
	int Cols = matImage.cols*matImage.channels();//x
	int x[100];
	int px = 0;
	int PixelDataof = 0;
	int sum = 0;
	double average = 0;
	for (int j = 0; j < Rows; j++) {
		uchar* data = matImage.ptr<uchar>(j);
  	for (int i = 0; i < Cols; i++) {
			PixelDataof = data[i];
			if (PixelDataof > 130) {
				x[px] = i;
				px++;
				sum = sum + i;
				if (px > 100) {
					cout << "there are too many canny points" << endl;
				}
			}
		}
		//逐行计算平均点
		if (px) {
			average = sum / px;
		}
		point[j].cx = average;
		point[j].cy = j;
		average = 0;
		sum = 0;
		px = 0;
		memset(x, 0, px);
	}
	//误差标记函数double
	getErrorIdentifyDouble(matImage,point);

	//cvWaitKey(0);
	//cvReleaseImage(&image);
	//cvReleaseImage(&pGrayImage);
	//cvReleaseImage(&pEdgeImage);
    //cvReleaseMat(&pGradMat);
	//cvReleaseMat(&pSubEdgeMatH);
	//cvReleaseMat(&pSubEdgeMatW);
	//cvReleaseMat(&pDirection);
}

//基于double的误差标记函数
void getErrorIdentifyDouble(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	double error;
	for (int j = 0; j < Rows; j++) {
		if (point[j].cx != point[j - 1].cx) {
			line(matImage, Point((point[j].cx - 30), point[j].cy), Point((point[j].cx + 30), point[j].cy), Scalar(255, 100, 100), 2, 8, 0);
			line(matImage, Point(point[j].cx, point[j].cy - 30), Point(point[j].cx, point[j].cy + 30), Scalar(255, 100, 100), 2, 8, 0);
			error = point[j].cx - point[j - 1].cx;
			ostringstream oss;
			oss << error;
			string texterror = oss.str();
			putText(matImage, texterror, Point(point[j].cx + 40, point[j].cy), 2, 0.5, Scalar(255, 100, 100), 1, 8, 0);
		}
	}
	namedWindow("error identification");
	imshow("error identification", matImage);
}

//双精度平均值计算
double average(int *x,int len) {
	int sum = 0;
	int num = 0;
	double average;
	for(int i = 0; i < len; i++ ){
		if (x[i]) {
			sum = sum + x[i];
			num++;
		}
	}
	average = sum / num;
	sum = 0;
	num = 0;
	return average;

}



//误差标记函数
void getErrorIdentifyInt(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	int error;
	for (int j = 0; j < Rows; j++) {
		if (point[j].x != point[j - 1].x) {
			line(matImage, Point((point[j].x - 30), point[j].y), Point((point[j].x + 30), point[j].y), Scalar(255, 100, 100), 2, 8, 0);
			line(matImage, Point(point[j].x, point[j].y - 30), Point(point[j].x, point[j].y + 30), Scalar(255, 100, 100), 2, 8, 0);
			error = point[j].x - point[j - 1].x;
			std::ostringstream oss;
			oss << error;
			std::string texterror = oss.str();
			putText(matImage, texterror,Point(point[j].x + 40, point[j].y),2 ,0.5, Scalar(255, 100, 100),1,8,0);
		}
	}
	namedWindow("error identification");
	imshow("error identification", matImage);
}

//被giveup的寻址method1
void getDiff1test(Mat matImage, MPoint *point) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	int MaxPixel;
	int error = 0;
	int errormax = 30;
	int errormin = 2;
	MPoint samepoint, startpoint;
	samepoint.x = 0;
	samepoint.y = 0;
	samepoint.bright = 0;
	samepoint.Pixnum = 0;
	for (int j = 1; j < Rows; j++) {

		if (((point[j].x) == (point[j - 1].x))) {
			;
		}
		else if ((point[j].x) != (point[j - 1].x) && !error) {   //第一次遇到转点
			error++;
			/*samepoint.x = point[j].x;
			samepoint.y = point[j].y;
			samepoint.bright = point[j].bright;
			samepoint.Pixnum = point[j].Pixnum;*/
			samepoint = point[j];
		}
		else if ((point[j].x != samepoint.x) && error) {
			error++;
		}
		else if (((point[j].x) == samepoint.x) && error) {
			//判断error是否符合标准
			if (error >= errormin && error <= errormax) {
				//cvLine画线标定
				line(matImage, Point((point[j].x - 30), point[j].y), Point((point[j].x + 30), point[j].y), Scalar(255, 100, 100), 2, 8, 0);
				line(matImage, Point(point[j].x, point[j].y - 30), Point(point[j].x, point[j].y + 30), Scalar(255, 100, 100), 2, 8, 0);
				line(matImage, Point((samepoint.x - 30), samepoint.y), Point((samepoint.x + 30), samepoint.y), Scalar(255, 100, 100), 2, 8, 0);
				line(matImage, Point(samepoint.x, samepoint.y - 30), Point(samepoint.x, samepoint.y + 30), Scalar(255, 100, 100), 2, 8, 0);
			}
			error = 0;
			samepoint.x = 0;
			samepoint.y = 0;
			samepoint.bright = 0;
			samepoint.Pixnum = 0;
		}
	}


	namedWindow("error identification");
	imshow("error identification", matImage);
}

//相机标定1
void calibfirst(Mat matImage) {
	ifstream fin("D:\\ChessTest\\calibdata.txt");//标定所用文件的路径
	ofstream fout("D:\\ChessTest\\calibresult.txt");//用于存储标定结果的文件
	cout << "开始提取角点" << endl;
	int image_count = 0;//图像数量
	Size image_size;//图像尺寸
	Size board_size = Size(10, 14);//标定板上每行每列的角点数
	vector <Point2f> image_points_buf;//缓存每幅图像上缓存的角点
	vector <vector<Point2f>> image_points_seq;//保存检测到的所有角点
	string filename;
	int count = -1; 
	while (getline(fin, filename)) {
		image_count++;
		cout << "image_count = " << image_count << endl;
		Mat imageinput = imread(filename);
		if (image_count == 1) {
			image_size.width = imageinput.cols;
			image_size.height = imageinput.rows;
		}

		//提取角点
		if (0 == findChessboardCorners(imageinput, board_size, image_points_buf))
		{
			cout << "cannot find chessboard corners!\n";
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageinput, view_gray, CV_RGB2GRAY);
			//亚像素精确化
			find4QuadCornerSubpix(view_gray,image_points_buf,Size(11,11));
			image_points_seq.push_back(image_points_buf);
			drawChessboardCorners(view_gray, board_size, image_points_buf, true);
			imshow("Camera Calibration", view_gray);
		}
	}
	int total = image_points_seq.size();
	cout << "total =" << total << endl;
	int CornerNum = board_size.width*board_size.height;
	for (int ii = 0; ii < total; ii++) {
		if (0 == ii % CornerNum)
		{
			int i = -1;
			i = ii / CornerNum;
			int j = i + 1;
			cout << "--->第" << j << "图片的数据--->" << endl;
		}
		if (0 == ii % 3)
		{
			cout << endl;
		}
		else
		{
			cout.width(10);
		}
		cout << "--->" << image_points_seq[ii][0].x;
		cout << "--->" << image_points_seq[ii][0].y;
	}
	cout << "角点提取完成！\n";
	//以下是摄像机标定
	cout << "开始标定……" << endl;
	//棋盘三维信息
	Size square_size = Size(18, 18);//实际测量得到的每个棋盘格的大小、
	vector<vector<Point3f>> object_points;//保存标定板上角点的三维坐标
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//摄像机内参数矩阵
	vector<int> point_counts;//每幅图上角点的数量
	Mat distCoeffs = Mat(3,3,CV_32FC1,Scalar::all(0));//摄像机的五个畸变系数：k1.k2,p1,p2,k3
	vector<Mat> tvecsMat;//每个图像的旋转向量
	vector<Mat> rvecsMat;//每个图像的平移向量
	//初始化标定板上角点的三维坐标
	int i, j, t;
	for (t = 0;t<image_count;t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++) {
			for (j = 0; j < board_size.width; j++) {
				Point3f realPoint;
				//假设标定板放在世界坐标系z=0的平面上
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	//初始化每个图像中的角点数量 假定每个图像中都可以看到完整的标定板
	for (i = 0; i < image_count; i++) {
		point_counts.push_back(board_size.width*board_size.height);
	}
	//开始标定
	calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs,rvecsMat,tvecsMat,0);
	cout << "开始评价评定结果……" << endl;
	double total_err = 0.8;
	double err = 0.0;
	vector<Point2f> image_points2;//保存重新计算得到的投影点
	fout << "每幅图像的标定误差：\n";
	for (i = 0; i < image_count; i++) {
		vector<Point3f> tempPointSet = object_points[1];
		//通过得到的摄像机内外参数对空间的三维点进行重新投影计算 得到新的投影点
		projectPoints(tempPointSet,rvecsMat[i],tvecsMat[i],cameraMatrix,distCoeffs,image_points2);
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1,image_points2.size(),CV_32FC2);
		for (int j = 0; j < tempImagePoint.size();j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x,image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat,tempImagePointMat,NORM_L2);
		total_err += err /= point_counts[i];
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	cout << "评价完成" << endl;
	//保存定标结果
	cout << "开始保存标定结果……" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs << endl << endl;
	for (int i = 0; i < image_count; i++) {
		fout<< "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << tvecsMat[i] << endl;
		/* 将旋转向量转换为相对应的旋转矩阵 */   		
		Rodrigues(tvecsMat[i], rotation_matrix);   		
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;   		
		fout << rotation_matrix << endl;   		
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;   		
		fout << rvecsMat[i] << endl << endl;
		
	}
	cout << "完成保存" << endl;
	fout << endl;
	//显示标定结果
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "保存校正后的图像" << endl;
	string imageFileName;
	stringstream StrStm;
	for (int i = 0; i != image_count; i++) {
		cout << "Frame #" << i + 1 << "..." << endl;
		//initUndis
	}
}
	

	

#ifdef USE_CALLBACK_GRAB_IMAGE
/*图像抓取回调函数*/
IplImage *g_iplImage = NULL;

void _stdcall GrabImageCallback(CameraHandle hCamera, BYTE *pFrameBuffer, tSdkFrameHead* pFrameHead, PVOID pContext)
{

	CameraSdkStatus status;


	//将获得的原始数据转换成RGB格式的数据，同时经过ISP模块，对图像进行降噪，边沿提升，颜色校正等处理。
	//我公司大部分型号的相机，原始数据都是Bayer格式的
	status = CameraImageProcess(hCamera, pFrameBuffer, m_pFrameBuffer, pFrameHead);

	//分辨率改变了，则刷新背景
	if (m_sFrInfo.iWidth != pFrameHead->iWidth || m_sFrInfo.iHeight != pFrameHead->iHeight)
	{
		m_sFrInfo.iWidth = pFrameHead->iWidth;
		m_sFrInfo.iHeight = pFrameHead->iHeight;
	}

	if (status == CAMERA_STATUS_SUCCESS)
	{
		//调用SDK封装好的显示接口来显示图像,您也可以将m_pFrameBuffer中的RGB数据通过其他方式显示，比如directX,OpengGL,等方式。
		CameraImageOverlay(hCamera, m_pFrameBuffer, pFrameHead);
		if (g_iplImage)
		{
			cvReleaseImageHeader(&g_iplImage);
		}
		g_iplImage = cvCreateImageHeader(cvSize(pFrameHead->iWidth, pFrameHead->iHeight), IPL_DEPTH_8U, /*sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? 1 :*/ 3);
		cvSetData(g_iplImage, m_pFrameBuffer, pFrameHead->iWidth*(/*sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? 1 : */3));
		cvShowImage(g_CameraName, g_iplImage);

		m_iDispFrameNum++;
		waitKey(30);//逐帧调取image
		//CLaserVisionDlg claservisiondlg;
		//claservisiondlg.CaptureImage(g_iplImage);
	}

	memcpy(&m_sFrInfo, pFrameHead, sizeof(tSdkFrameHead));

}

#else 
/*图像抓取线程，主动调用SDK接口函数获取图像*/
UINT WINAPI uiDisplayThread(LPVOID lpParam)
{
	tSdkFrameHead 	sFrameInfo;
	CameraHandle    hCamera = (CameraHandle)lpParam;
	BYTE*			pbyBuffer;
	CameraSdkStatus status;
	IplImage *iplImage = NULL;

	while (!m_bExit)
	{

		if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
		{
			//将获得的原始数据转换成RGB格式的数据，同时经过ISP模块，对图像进行降噪，边沿提升，颜色校正等处理。
			//我公司大部分型号的相机，原始数据都是Bayer格式的
			status = CameraImageProcess(hCamera, pbyBuffer, m_pFrameBuffer, &sFrameInfo);//连续模式

			//分辨率改变了，则刷新背景
			if (m_sFrInfo.iWidth != sFrameInfo.iWidth || m_sFrInfo.iHeight != sFrameInfo.iHeight)
			{
				m_sFrInfo.iWidth = sFrameInfo.iWidth;
				m_sFrInfo.iHeight = sFrameInfo.iHeight;
				//图像大小改变，通知重绘
			}

			if (status == CAMERA_STATUS_SUCCESS)
			{
				//调用SDK封装好的显示接口来显示图像,您也可以将m_pFrameBuffer中的RGB数据通过其他方式显示，比如directX,OpengGL,等方式。
				//CameraImageOverlay(hCamera, m_pFrameBuffer, &sFrameInfo);
#if 0
				if (iplImage)
				{
					cvReleaseImageHeader(&iplImage);
				}
				iplImage = cvCreateImageHeader(cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight), IPL_DEPTH_8U, sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? 1 : 3);
				cvSetData(iplImage, m_pFrameBuffer, sFrameInfo.iWidth*(sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? 1 : 3));
				cvShowImage(g_CameraName, iplImage);//展示初始捕获图像
				//对捕获图像进行处理		
				IplImage *imgOrign = 0;
				cvCopy(iplImage, imgOrign, NULL);
				IplImage *imgDest = 0;
				LaserRange laservision;
				LaserRange::RangeResult *temp = laservision.GetRange(imgOrign, imgDest);


				


#else
				cv::Mat matImage(
					cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight),
					sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
					m_pFrameBuffer
				);
				imshow(g_CameraName, matImage);
#endif

				int Rows = matImage.rows;//y
				//int Cols = matImage.cols;
				int Cols = matImage.cols*matImage.channels();//x
				MPoint *point;
				point = new MPoint[Rows];
#ifdef CALIB_CAMERA
				calibfirst(matImage);
#endif

#ifdef GET_CANNY
				
				//canny
				getcanny(matImage, point);
				//调用中心点finder
				//getPeaker1(matImage,point);
				//调用标记
				getErrorIdentifyDouble(matImage, point);
#endif

#ifdef GET_DOUBLE_PIXEL
				getdoublepixel(matImage,point);
#endif
				m_iDispFrameNum++;
				delete []point;
			}

			//在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
			//否则再次调用CameraGetImageBuffer时，程序将被挂起，知道其他线程中调用CameraReleaseImageBuffer来释放了buffer
			CameraReleaseImageBuffer(hCamera, pbyBuffer);
			memcpy(&m_sFrInfo, &sFrameInfo, sizeof(tSdkFrameHead));
		}

		int c = waitKey(10);

		if (c == 'q' || c == 'Q' || (c & 255) == 27)
		{
			m_bExit = TRUE;
			break;
		}
	}

	if (iplImage)
	{
		cvReleaseImageHeader(&iplImage);
	}

	_endthreadex(0);
		return 0;
#endif
}




int _tmain(int argc, _TCHAR* argv[])
{



	tSdkCameraDevInfo sCameraList[10];
	INT iCameraNums;
	CameraSdkStatus status;
	tSdkCameraCapbility sCameraInfo;

	//枚举设备，获得设备列表
	iCameraNums = 10;//调用CameraEnumerateDevice前，先设置iCameraNums = 10，表示最多只读取10个设备，如果需要枚举更多的设备，请更改sCameraList数组的大小和iCameraNums的值

	if (CameraEnumerateDevice(sCameraList, &iCameraNums) != CAMERA_STATUS_SUCCESS || iCameraNums == 0)
	{
		printf("No camera was found!");
		return FALSE;
	}

	//该示例中，我们只假设连接了一个相机。因此，只初始化第一个相机。(-1,-1)表示加载上次退出前保存的参数，如果是第一次使用该相机，则加载默认参数.
	//In this demo ,we just init the first camera.
	if ((status = CameraInit(&sCameraList[0], -1, -1, &m_hCamera)) != CAMERA_STATUS_SUCCESS)
	{
		char msg[128];
		sprintf_s(msg, "Failed to init the camera! Error code is %d", status);
		printf(msg);
		printf(CameraGetErrorString(status));
		return FALSE;
	}


	//Get properties description for this camera.
	CameraGetCapability(m_hCamera, &sCameraInfo);//"获得该相机的特性描述"

	m_pFrameBuffer = (BYTE *)CameraAlignMalloc(sCameraInfo.sResolutionRange.iWidthMax*sCameraInfo.sResolutionRange.iWidthMax * 3, 16);

	if (sCameraInfo.sIspCapacity.bMonoSensor)
	{
		CameraSetIspOutFormat(m_hCamera, CAMERA_MEDIA_TYPE_MONO8);
	}

	strcpy_s(g_CameraName, sCameraList[0].acFriendlyName);
	CameraSetMirror(m_hCamera,0,1);
	CameraSetRotate(m_hCamera,2);
	CameraCreateSettingPage(m_hCamera, NULL,
		g_CameraName, NULL, NULL, 0);//"通知SDK内部建该相机的属性页面";

#ifdef USE_CALLBACK_GRAB_IMAGE //如果要使用回调函数方式，定义USE_CALLBACK_GRAB_IMAGE这个宏
	//Set the callback for image capture
	CameraSetCallbackFunction(m_hCamera, GrabImageCallback, 0, NULL);//"设置图像抓取的回调函数";
#else
	m_hDispThread = (HANDLE)_beginthreadex(NULL, 0, &uiDisplayThread, (PVOID)m_hCamera, 0, &m_threadID);
#endif

	CameraPlay(m_hCamera);

	CameraShowSettingPage(m_hCamera, TRUE);//TRUE显示相机配置界面。FALSE则隐藏。

	while (m_bExit != TRUE)
	{
		waitKey(10);
	}

	CameraUnInit(m_hCamera);

	CameraAlignFree(m_pFrameBuffer);

	destroyWindow(g_CameraName);

#ifdef USE_CALLBACK_GRAB_IMAGE
	if (g_iplImage)
	{
		cvReleaseImageHeader(&g_iplImage);
	}
#endif
	return 0;
}

/*
int main() {
	UINT WINAPI uiDisplayThread(LPVOID lpParam);
	return 0;
}
*/



// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
