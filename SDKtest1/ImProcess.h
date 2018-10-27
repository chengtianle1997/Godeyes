#pragma once
#pragma once
//ImProcess.h
//2009-5-29����
#ifndef IMPROCESS_H
#define IMPROCESS_H

#define	PI			3.14159		
#define COUTINFO    1

#include <iostream>
#include "cxcore.h"		       //OpenCVͷ�ļ�
#include "highgui.h"
#include "cv.h" 

//#pragma comment (lib,"highgui.lib")
//#pragma comment (lib,"cxcore.lib")
////#pragma comment (lib,"cv.lib")

class CHTimer
{
public:
	CHTimer();
	~CHTimer();
	void StartTime();
	void EndTime();
	double GetTime();
protected:

private:
	long  m_lStartTime;
	long  m_lEndTime;
	long  m_lPersecond;
	double	  m_fTime;
};

int inline AbsInt(int a)
{
	if (a > 0)
		return a;
	else
		return -a;
}
double inline square(double a)
{
	return a * a;
}
//�˲��㷨
void hFilter(IplImage* src,   //Դͼ��
	IplImage* dst,   //Ŀ��ͼ�񣬼��˲����ͼ��
	int nThreshold); //��ֵ������ʱnThreshold=0������ʹ��Ĭ��ֵ�������Ը�ֵ��Ϊ��ֵ
//��Ե����㷨,sobel����
void hSobel(IplImage* src,    //ԭʼͼ��
	IplImage* dst,   //��Եͼ��
	CvMat* GradImage, //�ݶ�ͼ
	CvMat* Direction, //��ֱ�ڱ�Ե�ݶȵķ������
	int nThreshold1,  //��ֵ����ֵ1 nThreshold1<=nThreshold2
	int nThreshold2);//��ֵ����ֵ2 nThreshold1<=nThreshold2
void hSubPixelEdge(CvMat* GradImage,   //�Ҷ�ͼ��
	IplImage* EdgeImage,//��Եͼ��
	CvMat* direction,   //�ݶȷ������
	CvMat* SubEdgeMatH, //�����ر�ԵHֵ
	CvMat* SubEdgeMatW);//�����ر�ԵHֵ
#endif