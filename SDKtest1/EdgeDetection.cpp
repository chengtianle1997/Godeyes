

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
#include "EdgeDetection.h"

using namespace std;
using namespace cv;

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
	float step = 0.85;//����ֵ�ٷ���
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

void getPeaker(Mat matImage,int *brightness) {
	int Cols = matImage.cols;
	int Rows = matImage.rows;
	int PixelData;
	int MaxPixel = 0;
	for (int j = 0; j < Rows; j++) {
		uchar* data = matImage.ptr<uchar>(j);
		MaxPixel = data[0];
		for (int i = 1; i < Cols; i++) {
			if (data[i] > MaxPixel) {
				MaxPixel = data[i];
			}
		}
		brightness[j] = MaxPixel;
	}
}

//Canny opencv ��Ե���
void getcanny(Mat matImage, MPoint *point) {
	Mat cloneImage = matImage.clone();
	int g_nCannyLowThreshold = 100;//canny������ֵ
	Mat tmpImage, dstImage;
	blur(cloneImage, tmpImage, Size(3, 3));
	Canny(tmpImage, dstImage, g_nCannyLowThreshold, g_nCannyLowThreshold * 3);
	namedWindow("canny function");
	imshow("canny function", dstImage);
	int Rows = cloneImage.rows;
	int Cols = cloneImage.cols*cloneImage.channels();
	int x[100];
	int px = 0;
	int PixelDataof;
	int sum = 0;
	double average = 0;
	for (int j = 0; j < Rows; j++) {
		uchar* data = dstImage.ptr<uchar>(j);
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
		//���м���ƽ����
		if (px) {
			  
			average = sum * 1.0 / px;
		}
		point[j].cx = average;
		point[j].cy = j;
		average = 0;
		sum = 0;
		px = 0;
		memset(x, 0, px);
		//cout << "(" << point[j].cx << "," << point[j].cy << ")" << endl;
	}
	getErrorIdentifyDouble(cloneImage, point);
	cloneImage.release();
}
//Sobel��Ե���
void getsobel(Mat matImage, MPoint *point) {
	Mat cloneImage = matImage.clone();
	Mat tempImage, tempImage1, tempImage2, tempImage_gauss;
	int ddepth = CV_16S, dx, dy,
		p_ksize = 3,//CV_SCHARR,   //�ں� -1�൱��3*3�ں���schar����
		p_scale = 2,		//���㵼��ʱ��������  ���ã��� �����Ժ� �������ָ��ף�����ɢ��
		p_delta = 0,	//�����ӵ������ĳ��� �������Ժ� ���������һ� �������ˣ���
		p_borderType = BORDER_DEFAULT;

	//16ת8λͼ  �����޸�
	double p_alpha_to8 = 1.0;// ��������
	double p_beta_to8 = 0.0; // ƫ����

	//��Ȩ
	double p_alpha_weig_x = 0.9, p_delta_weig_y = 0.1;//xyȨ��
	double p_gamma_weig = 0; //��Ȩ��ƫ����
	dx = 1;
	dy = 0;


	//û�д���
	GaussianBlur(cloneImage, tempImage_gauss, Size(7, 7), 0, 0, BORDER_DEFAULT);
	/*namedWindow("gauss");
	imshow("gauss", matImage);*/

	Sobel(tempImage_gauss, tempImage, ddepth, dx, dy, p_ksize, p_scale, p_delta, p_borderType);
	convertScaleAbs(tempImage, tempImage1, p_alpha_to8, p_beta_to8);

	dx = 0;
	dy = 1;
	Sobel(tempImage_gauss, tempImage, ddepth, dx, dy, p_ksize, p_scale, p_delta, p_borderType);
	convertScaleAbs(tempImage, tempImage2, p_alpha_to8, p_beta_to8);

	addWeighted(tempImage1, p_alpha_weig_x, tempImage2, p_delta_weig_y, p_gamma_weig, tempImage, -1);

	namedWindow("sobel");
	imshow("sobel", tempImage);

	//cvtColor(tempImage, tempImage, CV_RGB2GRAY);
	//tempImageΪ��Եͼ
	int nr = tempImage.rows;
	int nc = tempImage.cols*tempImage.channels();
	int PixelMin_Range = 80;
	//int PixelMax_Range;
	//int PixelMax=80;
	int n_point = 0;
	int x_sum = 0;
	int Pixel_temp;


	for (int j = 0; j < nr; j++)
	{
		uchar* data = tempImage.ptr<uchar>(j);
		//	PixelMax = data[0];
		for (int i = 0; i < nc; i++)
		{
			Pixel_temp = data[i];
			if (Pixel_temp > PixelMin_Range)
			{
				x_sum = x_sum + i;
				n_point++;
			}
		}
		point[j].cx = x_sum * 1.0 / n_point;
		point[j].cy = j;
		x_sum = 0;
		n_point = 0;
	}
	getErrorIdentifyDoubleW(cloneImage, point, 0.5);
	cloneImage.release();
}

//Laplician ��Ե���
void getlaplacian(Mat matImage, MPoint *point)
{
	Mat cloneImage = matImage.clone();
	Mat tempImage_gauss, tempImage1, tempImage2;
	Mat new_image = Mat::zeros(cloneImage.size(), cloneImage.type());
	int pl_ddepth = CV_16S,
		pl_kernel_size = 3,
		pl_scale = 1,
		pl_delta = 0;
	GaussianBlur(cloneImage, tempImage_gauss, Size(7, 7), 0, 0, BORDER_DEFAULT);
	//namedWindow("gauss");
	//imshow("gauss", tempImage_gauss);


	//cvtColor(tempImage_gauss, tempImage_gauss, CV_RGB2GRAY);
	Laplacian(tempImage_gauss, tempImage1, pl_ddepth, pl_kernel_size, pl_scale, pl_delta, BORDER_DEFAULT);
	convertScaleAbs(tempImage1, tempImage2);

	/*double alpha_co =5;
	double beta_co = 0;
	tempImage2.convertTo(new_image, -1, alpha_co, beta_co);
	tempImage2=new_image.clone();
	new_image.release();

*/

	namedWindow("laplacian");
	imshow("laplacian", tempImage2);

	int nr = tempImage2.rows,
		nc = tempImage2.cols*tempImage2.channels();

	//	cout << nr << " " << nc;
	int PixelMin = 20;
	int x_sum = 0;
	int n_point = 0;
	int max;
	//nr = 1;
	for (int j = 0; j < nr; j++)
	{
		uchar *data = tempImage2.ptr<uchar>(j);
		max = data[0];
		for (int i = 0; i < nc; i++)
		{
			if (data[i] > max)
				max = data[i];
				//cout << (int)data[i] << " ";
			if (data[i] > PixelMin)
			{
				x_sum += i;
				n_point++;
			}
		}
		//cout << max<< endl;
		if (n_point == 0) {
			n_point = 1;
		}
		point[j].cx = x_sum / n_point;
		point[j].cy = j;

		//cout << "(" << point[j].cx << "," << point[j].cy << ")" << endl;
		x_sum = 0;
		n_point = 0;
	}

	getErrorIdentifyDouble(cloneImage, point);
	cloneImage.release();
	//waitKey(0);

}


//�����ط��������������ƣ�
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
	//canny��Ե�㷨
	cvCanny(pGrayImage, pEdgeImage, 120, 180);
	//sobel��Ե�㷨
	hSobel(pGrayImage, pEdgeImage, pGradMat, pDirection, 10, 120);
	//�����ط���
	hSubPixelEdge(pGradMat, pEdgeImage, pDirection, pSubEdgeMatH, pSubEdgeMatW);
	//timer.EndTime();
	//cout << "time:" << timer.GetTime() << "ms" << endl;
	cvShowImage("Image", pEdgeImage);

	////�����ɵ�ͼ����н���	
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
		//���м���ƽ����
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
	//����Ǻ���double
	getErrorIdentifyDouble(matImage, point);

	//cvWaitKey(0);
	//cvReleaseImage(&image);
	//cvReleaseImage(&pGrayImage);
	//cvReleaseImage(&pEdgeImage);
	//cvReleaseMat(&pGradMat);
	//cvReleaseMat(&pSubEdgeMatH);
	//cvReleaseMat(&pSubEdgeMatW);
	//cvReleaseMat(&pDirection);
}

//Zernike�������ط���
//

//���ڸ�˹��ϵ������������߼���㷨  
void getGaussCenter(Mat matImage, MPoint *point,double maxError,double minError,int xRange) {
	Mat cloneImage = matImage.clone();
	Mat OrgnImage = matImage.clone();
	//������canny���õ�����������
	int g_nCannyLowThreshold = 100;//canny������ֵ
	Mat tmpImage, dstImage;
	blur(cloneImage, tmpImage, Size(3, 3));
	Canny(tmpImage, dstImage, g_nCannyLowThreshold, g_nCannyLowThreshold * 3);
	namedWindow("canny function");
	imshow("canny function", dstImage);
	int Rows = cloneImage.rows;
	int Cols = cloneImage.cols*cloneImage.channels();
	int x[100];
	int px = 0;
	int PixelDataof;
	int *brightness;
	brightness = new int[Rows];
	memset(brightness, 0, Rows);
	int sum = 0;
	double average = 0;
	getPeaker(matImage, brightness);
	for (int j = 0; j < Rows; j++) {
		uchar* data = dstImage.ptr<uchar>(j);
		for (int i = 0; i < Cols; i++) {
			PixelDataof = data[i];
			if (PixelDataof > minError*brightness[j]) {
				x[px] = i;
				px++;
				sum = sum + i;
				if (px > 100) {
					cout << "there are too many canny points" << endl;
				}
			}
		}
		//���м���ƽ����
		if (px) {
			average = sum * 1.0 / px;
		}
		point[j].cx = average;
		point[j].cy = j;
		
		average = 0;
		sum = 0;
		px = 0;
		memset(x, 0, px);
		//cout << "(" << point[j].cx << "," << point[j].cy << ")" << endl;
	}
	
	//��ȡpoint�е�ֵ
//	int Cols = cloneImage.cols;//x
//	int Rows = cloneImage.rows;//y
	
	int PixelData;
	int Pixnum = 0;
	
	GPoint *gpoint;
	gpoint = new GPoint[Rows];
	//���д洢���е��x���������ֵ�Ա���� �ڴ�ֻ�����˹��
	for (int i = 0; i < Rows; i++) {
		Pixnum = 0;
		uchar* data = matImage.ptr<uchar>(i);
		for (int j = 0; j < Cols; j++) {
			PixelData = data[j];
			//cout << PixelData << endl;
			//minerror��maxerror����ɸѡ��˹��  //�����ڴ˴�����xRange
			//cout << "condition1" << (PixelData > minError*brightness[i]) << endl;
			//cout << "condition2" << (PixelData < ((1 - maxError)*brightness[i]))<<endl;
			//cout << "condition3" << (abs(j - point[i].x) < xRange) << endl;

			if (PixelData > minError*brightness[i] && PixelData < ((1 - maxError)*brightness[i])&&(abs(j-point[i].cx)<xRange)) {
					gpoint[j].x = j;
					gpoint[j].brightness = PixelData;
					Pixnum++;
			}
		}
		int n, m;
		double *x, *y, *a;
		n = Pixnum;//������ݵ�ĸ���
		m = Pixnum; //��϶���ʽ������ m<=n
		x = new double[n];
		y = new double[n];
		a = new double[m];
		for (int i = 0; i < n; i++) {
			x[i] = gpoint[i].x;
			y[i] = gpoint[i].brightness;
		}  
		MinDoubleFit(x, y, n, a, m + 1);
		cout << "��϶���ʽ��ϵ��Ϊ" << endl;
		for (int i = 0; i <= m; i++) {
			cout << a[i] << '    ';
		}
		cout << endl;
		cout << "ƽ�����Ϊ��" << endl;
		cout << ErrorSqrt(x, y, n, a, m + 1) << endl;
	}
	delete brightness;
}

//f(n,x)��������x��n�η�
double f(int n, double x) {
	double y = 1.0;
	if (n == 0) return 1.0;
	else {
		for (int i = 0; i < n; i++) {
			y*=x;
		}
		return y;
	}
}

//��˹��Ԫ����ⷽ����
int GaussMajorEquation(double **a, int n, double *b, double *p, double dt) {
	int i, j, k, l;
	double c, t;
	for (k = 1; k <= n; k++) {
		c = 0.0;
		for (i = k; i <= n; i++) {
			if (fabs(a[i - 1][k - 1] > fabs(c))) {
				c = a[i - 1][k - 1];
				l = i;
			}
			if (fabs(c) <= dt)
				return 0;
			if (l != k) {
				for (j = k; j <= n; j++) {
					t = a[k - 1][j - 1];
					a[k - 1][j - 1] = a[l - 1][j - 1];
					a[l - 1][j - 1] = t;
				}
				t = b[k - 1];
				b[k - 1] = b[l - 1];
				b[l - 1] = t;
			}
			c = 1 / c;
			for (j = k + 1; j <= n; j++) {
				a[k - 1][j - 1] = a[k - 1][j - 1] * c;
				for (i = k + 1; i <= n; i++) {
					a[i - 1][j - 1] -= a[i - 1][k - 1] * a[k - 1][j - 1];
				}
			}
			b[k - 1] *= c;
			for (i = k + 1; i <= n; i++) {
				b[i - 1] -= b[k - 1] * a[i - 1][k - 1];
			}

			for (i = n; i >= 1; i--) {
				for (j = i + 1; j <= n; j++) {
					b[i - 1] -= b[j - 1] * a[i - 1][j - 1];
				}
			}
			cout.precision(12);
			for (i = 0; i < n; i++)
				p[i] = b[i];
		}
	}
}

//��̬��������
double** create(int a, int b) {
	double **p = new double* [a];
	for (int i = 0; i < b; i++) {
		p[i] = new double[b];
	}
	return p;
}

//��С�������//    m A����Ĵ�С
void MinDoubleFit(double x[], double y[], int n, double a[], int m) {
	int i, j, k, l;
	double **A, *B;
	A = create(m, m);
	B = new double[m];
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			A[i][j] = 0.0;
		}
	}
	for (k = 0; k < m; k++) {
		for (l = 0; l < m; l++) {
			for (j = 0; j < n; j++) {
				A[k][l] += f(k, x[j])*f(l, x[j]);
			}
		}
	}//���㷨������ϵ������A[k][l]
	cout << "���������ϵ������Ϊ��" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0, k = 1; j < m; j++, k++) {
			cout << A[i][j] << '\t';
			if (k&&k%m == 0) {
				cout << endl;
			}
		}
	}
	for (i = 0; i < m; i++) {
		B[i] = 0.0;
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			B[i] += y[i] * f(i, x[j]);
		}
	}
	//��¼B[n]
	for (i = 0; i < m; i++) {
		cout << "B[" << i << "]=" << B[i] << endl;
	}
	GaussMajorEquation(A,m,B,a,1e-6);
	delete[]A;
	delete B;
}

//������С���˽��ƽ�����
double ErrorSqrt(double x[], double y[], int n, double a[], int m) {
	double deta, q = 0.0, r = 0.0;
	int i, j;
	double *B;
	B = new double[m];
	for (i = 0; i < m; i++)
		B[i] = 0.0;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			B[i] += y[j] * f(i, x[j]);
	for (i = 0; i < n; i++)
		q += y[i] * y[i];
	for (j = 0; j < m; j++)
		r += a[j] * B[j];
	deta = fabs(q - r);
	return deta;
	delete B;
}


//����double������ֵ����Ǻ���
void getErrorIdentifyDoubleW(Mat matImage, MPoint *point, double doorin) {
	int Rows = matImage.rows;//y
	//int Cols = matImage.cols;
	int Cols = matImage.cols*matImage.channels();//x
	//int div = 64;
	double error;
	for (int j = 0; j < Rows; j++) {
		if (abs(point[j].cx - point[j - 1].cx) > doorin) {
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

//����double������Ǻ���
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

//˫����ƽ��ֵ����
double average(int *x, int len) {
	int sum = 0;
	int num = 0;
	double average;
	for (int i = 0; i < len; i++) {
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



//����Ǻ���
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
			putText(matImage, texterror, Point(point[j].x + 40, point[j].y), 2, 0.5, Scalar(255, 100, 100), 1, 8, 0);
		}
	}
	namedWindow("error identification");
	imshow("error identification", matImage);
}

//��giveup��Ѱַmethod1
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
		else if ((point[j].x) != (point[j - 1].x) && !error) {   //��һ������ת��
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
			//�ж�error�Ƿ���ϱ�׼
			if (error >= errormin && error <= errormax) {
				//cvLine���߱궨
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
