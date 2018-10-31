#pragma once
#include "pch.h"
#include <iostream>
#include "stdafx.h"
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


using namespace std;
using namespace cv;

typedef struct GPoint {
	int x;
	int brightness;
} GPoint;

//��Ե�����㷨1
void getPeaker1(Mat matImage, MPoint *point);
//Canny opencv ��Ե���
void getcanny(Mat matImage, MPoint *point);
//laplician ��Ե���
void getlaplacian(Mat matImage, MPoint *point);
//Sobel ��Ե���
void getsobel(Mat matImage, MPoint *point);
//�����ط��������������ƣ�
void getdoublepixel(Mat matImage, MPoint *point);
//���ڸ�˹��ϵ������������߼���㷨  
void getGaussCenter(Mat matImage, MPoint *point, double maxError, double minError, int xRange);
//X,Z��������� //X���� Z����  ���ݵ����  ����GPoint
int getXZmatrix(CvMat* X, CvMat* Z, int n, GPoint *gpoint);
//f(n,x)��������x��n�η�
double f(double x, int n);
//��˹��Ԫ����ⷽ����
int GaussMajorEquation(double **a, int n, double *b, double *p, double dt);
//��̬��������
double** create(int a, int b);
//��С�������//    m A����Ĵ�С
void MinDoubleFit(double x[], double y[], int n, double a[], int m);
//������С���˽��ƽ�����
double ErrorSqrt(double x[], double y[], int n, double a[], int m);
//����double�Ĵ���ֵ����Ǻ���
void getErrorIdentifyDoubleW(Mat matImage, MPoint *point, double doorin,int eHeight);
//����double������Ǻ���
void getErrorIdentifyDouble(Mat matImage, MPoint *point);
//˫����ƽ��ֵ����
double average(int *x, int len);
//����Ǻ���
void getErrorIdentifyInt(Mat matImage, MPoint *point);
//��giveup��Ѱַmethod1
void getDiff1test(Mat matImage, MPoint *point);