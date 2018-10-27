#include "pch.h"
#include "iostream"
#include "cxcore.h"		//OpenCVͷ�ļ�
#include "highgui.h"
#include "cv.h" 
#include "ImProcess.h"
#include "windows.h"

using namespace std;
using namespace cv;

//***************************************************
CHTimer::CHTimer()
{
	m_lStartTime = 0;
	m_lEndTime = 0;
	m_lPersecond = 0;
	m_fTime = 0.0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&m_lPersecond);//ѯ��ϵͳһ���ӵ�Ƶ��
}

//***************************************************
CHTimer::~CHTimer()
{

}

//***************************************************
//������ʱ
void CHTimer::StartTime()
{
	QueryPerformanceCounter((LARGE_INTEGER *)&m_lStartTime);
}

//***************************************************
//ֹͣ��ʱ
void CHTimer::EndTime()
{
	QueryPerformanceCounter((LARGE_INTEGER *)&m_lEndTime);
}

//***************************************************
//��ü�ʱʱ��(ms),�Ժ���Ϊ��λ
double CHTimer::GetTime()
{
	m_fTime = (double)(m_lEndTime - m_lStartTime) / m_lPersecond;
	m_fTime = m_fTime * 1000;
	return m_fTime;
}

//***************************************************
//�˲��㷨
void hFilter(IplImage* src,   //Դͼ��
	IplImage* dst,   //Ŀ��ͼ�񣬼��˲����ͼ��
	int nThreshold)  //��ֵ������ʱnThreshold=0������ʹ��Ĭ��ֵ�������Ը�ֵ��Ϊ��ֵ
{
	int nWidth = 0, nHeight = 0;
	char* pSrcData = NULL;
	char* pDstData = NULL;
	int nSrcStep = 0, nDstStep = 0;
	int h = 0, w = 0, nh = 0, nw = 0;
	int nPA[9], nSum1 = 0, nSum2 = 0, nMax = 0, nTemp = 0;
	nWidth = src->width - 1;
	nHeight = src->height - 1;
	nSrcStep = src->widthStep;
	nDstStep = dst->widthStep;
	pSrcData = src->imageData;
	pDstData = dst->imageData;

	if (nThreshold == 0)
		nThreshold = 30;
	//�Ա�Ե��Ĵ���,Ϊԭֵ
	for (w = 0; w < nWidth + 1; w++)
	{
		((uchar*)pDstData)[w] = ((uchar*)pSrcData)[w];
		((uchar*)(pDstData + nHeight * nDstStep))[w] = ((uchar*)(pSrcData + nHeight * nSrcStep))[w];
	}
	for (h = 0; h < nHeight + 1; h++)
	{
		((uchar*)(pDstData + h * nDstStep))[0] = ((uchar*)(pSrcData + h * nSrcStep))[0];
		((uchar*)(pDstData + h * nDstStep))[nWidth] = ((uchar*)(pSrcData + h * nSrcStep))[nWidth];
	}
	for (h = 1; h < nHeight; h++)
	{
		for (w = 1; w < nWidth; w++)
		{
			nPA[0] = ((uchar*)(pSrcData + (h - 1)*nSrcStep))[w - 1];
			nPA[1] = ((uchar*)(pSrcData + (h - 1)*nSrcStep))[w];
			nPA[2] = ((uchar*)(pSrcData + (h - 1)*nSrcStep))[w + 1];
			nPA[3] = ((uchar*)(pSrcData + h * nSrcStep))[w - 1];
			nPA[4] = ((uchar*)(pSrcData + h * nSrcStep))[w];
			nPA[5] = ((uchar*)(pSrcData + h * nSrcStep))[w + 1];
			nPA[6] = ((uchar*)(pSrcData + (h + 1)*nSrcStep))[w - 1];
			nPA[7] = ((uchar*)(pSrcData + (h + 1)*nSrcStep))[w];
			nPA[8] = ((uchar*)(pSrcData + (h + 1)*nSrcStep))[w + 1];
			nSum1 = nPA[0] + nPA[3] + nPA[6];
			nSum2 = nPA[2] + nPA[5] + nPA[8];
			nMax = AbsInt(nSum1 - nSum2);
			nSum1 = nPA[0] + nPA[1] + nPA[3];
			nSum2 = nPA[5] + nPA[7] + nPA[8];
			nTemp = AbsInt(nSum1 - nSum2);
			if (nTemp > nMax)
				nMax = nTemp;
			nSum1 = nPA[0] + nPA[1] + nPA[2];
			nSum2 = nPA[6] + nPA[7] + nPA[8];
			nTemp = AbsInt(nSum1 - nSum2);
			if (nTemp > nMax)
				nMax = nTemp;
			nSum1 = nPA[1] + nPA[2] + nPA[5];
			nSum2 = nPA[3] + nPA[6] + nPA[7];
			nTemp = AbsInt(nSum1 - nSum2);
			if (nTemp > nMax)
				nMax = nTemp;
			if (nMax < nThreshold)//С����ֵ����ǰ��ĻҶȣ�Ϊ��3��3�����ƽ��ֵ
			{
				nTemp = nSum1 + nSum2 + nPA[0] + nPA[4] + nPA[8];
				((uchar*)(pDstData + h * nDstStep))[w] = nTemp / 9;
			}
			else				//���򣬵�ǰ��ĻҶȣ�Ϊ��3��3������ֵ
			{
				for (nw = 0; nw < 5; nw++)
				{
					for (nh = 0; nh < 8 - nw; nh++)
					{
						if (nPA[nh] > nPA[nh + 1])
						{
							nTemp = nPA[nh];
							nPA[nh] = nPA[nh + 1];
							nPA[nh + 1] = nTemp;
						}
					}
				}
				((uchar*)(pDstData + h * nDstStep))[w] = nPA[4];
			}
		}
	}
}

//***************************************************
//��Ե����㷨,sobel����
void hSobel(IplImage* src,    //ԭʼͼ��
	IplImage* dst,   //��Եͼ��
	CvMat* GradImage, //�ݶ�ͼ
	CvMat* pFlagMat, //��ֱ�ڱ�Ե�ݶȵķ������
	int nThreshold1,  //��ֵ����ֵ1 nThreshold1<=nThreshold2
	int nThreshold2) //��ֵ����ֵ2 nThreshold1<=nThreshold2
{
	CvMat* pEdgeMat = NULL;
	int* pMatData = NULL;
	char* pSrcData = NULL;
	char* pDstData = NULL;
	unsigned char* pFlagData = NULL;
	float* pDirect = NULL;
	int nSrcStep = 0, nDstStep = 0, nMatStep = 0, nDirectStep = 0, nFlagStep = 0;
	int h = 0, w = 0, nh = 0, nw = 0;
	int nPA[9], nSum1 = 0, nSum2 = 0, nMax = 0, nTemp = 0;
	int Hvalue, Vvalue, Lvalue, mLvalue, nDFlag = 0, nMaxFlag = 0;
	int nWidth = 0, nHeight = 0;

	pEdgeMat = cvCreateMat(src->height, src->width, CV_32SC1);
	pMatData = pEdgeMat->data.i;
	nMatStep = (pEdgeMat->step) / sizeof(int);
	nWidth = src->width - 1;
	nHeight = src->height - 1;
	nSrcStep = src->widthStep;
	nDstStep = dst->widthStep;
	pSrcData = src->imageData;
	pDstData = dst->imageData;
	pFlagData = pFlagMat->data.ptr;
	nFlagStep = pFlagMat->step;
	//���Ե����
	for (h = 1; h < nHeight; h++)
	{
		for (w = 1; w < nWidth; w++)
		{
			nPA[0] = ((uchar*)(pSrcData + (h - 1)*nSrcStep))[w - 1];
			nPA[1] = ((uchar*)(pSrcData + (h - 1)*nSrcStep))[w];
			nPA[2] = ((uchar*)(pSrcData + (h - 1)*nSrcStep))[w + 1];
			nPA[3] = ((uchar*)(pSrcData + h * nSrcStep))[w - 1];
			nPA[4] = ((uchar*)(pSrcData + h * nSrcStep))[w];
			nPA[5] = ((uchar*)(pSrcData + h * nSrcStep))[w + 1];
			nPA[6] = ((uchar*)(pSrcData + (h + 1)*nSrcStep))[w - 1];
			nPA[7] = ((uchar*)(pSrcData + (h + 1)*nSrcStep))[w];
			nPA[8] = ((uchar*)(pSrcData + (h + 1)*nSrcStep))[w + 1];
			nSum1 = nPA[0] + nPA[3] + nPA[3] + nPA[6];
			nSum2 = nPA[2] + nPA[5] + nPA[5] + nPA[8];
			nMax = AbsInt(nSum1 - nSum2);
			Vvalue = nMax;
			nDFlag = 1;//�����н�Ϊ0��(������Ϊ����ָ����)
			nSum1 = nPA[0] + nPA[0] + nPA[1] + nPA[3];
			nSum2 = nPA[5] + nPA[7] + nPA[8] + nPA[8];
			nTemp = AbsInt(nSum1 - nSum2);
			Lvalue = nTemp;
			if (nTemp > nMax)
			{
				nMax = nTemp;
				nDFlag = 2;//�����н�Ϊ��45��(������Ϊ����ָ���ң���ʱ��Ϊ��)
			}
			nSum1 = nPA[0] + nPA[1] + nPA[1] + nPA[2];
			nSum2 = nPA[6] + nPA[7] + nPA[7] + nPA[8];
			nTemp = AbsInt(nSum1 - nSum2);
			Hvalue = nTemp;
			if (nTemp > nMax)
			{
				nMax = nTemp;
				nDFlag = 3;//�����н�Ϊ��90��(��������ͬ��)(������Ϊ����ָ���ң���ʱ��Ϊ��)
			}
			nSum1 = nPA[1] + nPA[2] + nPA[2] + nPA[5];
			nSum2 = nPA[3] + nPA[6] + nPA[6] + nPA[7];
			nTemp = AbsInt(nSum1 - nSum2);
			mLvalue = nTemp;
			if (nTemp > nMax)
			{
				nMax = nTemp;
				nDFlag = 4;//�����н�Ϊ��135��(��������ͬ��)(������Ϊ����ָ���ң���ʱ��Ϊ��)
			}
			(pMatData + h * nMatStep)[w] = nMax;
			(GradImage->data.i + h * (GradImage->step / sizeof(int)))[w] = nMax;
			(pFlagData + h * nFlagStep)[w] = nDFlag;
		}
	}
	//�Ǽ���ֵ����
	for (h = 1; h < nHeight; h++)
	{
		for (w = 1; w < nWidth; w++)
		{
			nDFlag = (pFlagData + h * nFlagStep)[w];
			nMax = (pMatData + h * nMatStep)[w];
			nMaxFlag = 0;
			((uchar*)(pDstData + h * nDstStep))[w] = 0;
			(pMatData + h * nMatStep)[w] = 0;
			switch (nDFlag)
			{
			case 1:
				if (nMax > (pMatData + h * nMatStep)[w - 1])
					if (nMax > (pMatData + h * nMatStep)[w + 1])nMaxFlag = 1;
				break;
			case 2:
				if (nMax > (pMatData + (h - 1)*nMatStep)[w - 1])
					if (nMax > (pMatData + (h + 1)*nMatStep)[w + 1])nMaxFlag = 1;
				break;
			case 3:
				if (nMax > (pMatData + (h - 1)*nMatStep)[w])
					if (nMax > (pMatData + (h + 1)*nMatStep)[w])nMaxFlag = 1;
				break;
			case 4:
				if (nMax > (pMatData + (h - 1)*nMatStep)[w + 1])
					if (nMax > (pMatData + (h + 1)*nMatStep)[w - 1])nMaxFlag = 1;
				break;
			default:
				break;
			}
			if (nMaxFlag == 1)
			{
				(pMatData + h * nMatStep)[w] = nMax;
				if (nMax > 255)
					((uchar*)(pDstData + h * nDstStep))[w] = 255;
				else
					((uchar*)(pDstData + h * nDstStep))[w] = nMax;
			}
		}
	}

	//˫��ֵ��ֵ��
	if (nThreshold1 > nThreshold2)//��֤nThreshold2>nThreshold1
	{
		nTemp = nThreshold1;
		nThreshold1 = nThreshold2;
		nThreshold2 = nTemp;
	}
	for (w = 0; w < nWidth + 1; w++)//�Ա�Ե��Ĵ���,Ϊ0
	{
		((uchar*)pDstData)[w] = 0;
		((uchar*)(pDstData + nHeight * nDstStep))[w] = 0;
	}
	for (h = 0; h < nHeight + 1; h++)
	{
		((uchar*)(pDstData + h * nDstStep))[0] = 0;
		((uchar*)(pDstData + h * nDstStep))[nWidth] = 0;
	}
	for (h = 1; h < nHeight; h++)
	{
		for (w = 1; w < nWidth; w++)
		{
			nMax = ((uchar*)(pDstData + h * nDstStep))[w];
			if (nMax > nThreshold2)
				((uchar*)(pDstData + h * nDstStep))[w] = 255;
			else if (nMax < nThreshold1)
				((uchar*)(pDstData + h * nDstStep))[w] = 0;
			else
			{
				nDFlag = 0;
				for (nh = -1; nh <= 1; nh++)
				{
					for (nw = -1; nw <= 1; nw++)
					{
						if (((uchar*)(pDstData + (h + nh)*nDstStep))[w | nw] > nThreshold2)
						{
							nDFlag = 1;
							nh = 3;     //����˫��ѭ��
							break;
						}
					}
				}
				if (nDFlag) ((uchar*)(pDstData + h * nDstStep))[w] = 255;
				else ((uchar*)(pDstData + h * nDstStep))[w] = 0;
			}
		}
	}
	cvReleaseMat(&pEdgeMat);
}

//***************************************************
//���ڶ���ʽ�������ر�Ե���
void hSubPixelEdge(CvMat* GradImage,   //�Ҷ�ͼ��
	IplImage* EdgeImage,//��Եͼ��
	CvMat* direction,   //�ݶȷ������
	CvMat* SubEdgeMatH, //�����ر�ԵHֵ
	CvMat* SubEdgeMatW) //�����ر�ԵHֵ
{
	int h = 0, w = 0, nWidth = 0, nHeight = 0, nEdgeWidthStep = 0, nGradWidthStep, nDirWidthStep = 0;
	int* pGradImage = NULL;
	char* pEdgeImage = NULL;
	unsigned char* pDirection = NULL;
	int nG0 = 0, nG1 = 0, nG2 = 0;
	char dir = 0;
	double fEdge = 0.0, fEdgeH = 0.0, fEdgeW = 0.0;

	nWidth = GradImage->width;
	nHeight = GradImage->height;

	pGradImage = GradImage->data.i;
	pEdgeImage = EdgeImage->imageData;
	pDirection = direction->data.ptr;

	nEdgeWidthStep = EdgeImage->widthStep;
	nGradWidthStep = GradImage->step / (sizeof(int));
	nDirWidthStep = direction->step / (sizeof(char));

	for (h = 1; h < nHeight; h++)
	{
		for (w = 1; w < nWidth; w++)
		{
			fEdgeH = 0.0;
			fEdgeW = 0.0;
			if (((uchar*)(pEdgeImage + h * nEdgeWidthStep))[w] == 255)
			{
				dir = ((uchar*)(pDirection + h * nDirWidthStep))[w];
				switch (dir)
				{
				case 1:
					nG0 = (pGradImage + h * nGradWidthStep)[w - 1];
					nG1 = (pGradImage + h * nGradWidthStep)[w];
					nG2 = (pGradImage + h * nGradWidthStep)[w + 1];
					fEdgeH = h;
					fEdgeW = w + ((double)(nG0 - nG2) / (nG0 + nG2 - 2 * nG1))*0.5;
					//cout<<"h:"<<h<<"  w:"<<w<<"  fw:"<<fEdgeW<<"  "<<nG0<<" "<<nG2<<" "<<nG1<<endl;
					break;
				case 2:
					nG0 = (pGradImage + (h - 1)*nGradWidthStep)[w - 1];
					nG1 = (pGradImage + h * nGradWidthStep)[w];
					nG2 = (pGradImage + (h + 1)*nGradWidthStep)[w + 1];
					fEdgeH = h + ((double)(nG0 - nG2) / (nG0 + nG2 - 2 * nG1))*0.5;
					fEdgeW = w + ((double)(nG0 - nG2) / (nG0 + nG2 - 2 * nG1))*0.5;
					//cout<<"h:"<<h<<"  w:"<<w<<"  fh:"<<fEdgeH<<"  fw:"<<fEdgeW<<"   "<<nG0<<" "<<nG2<<" "<<nG1<<endl;
					break;
				case 3:
					nG0 = (pGradImage + (h - 1)*nGradWidthStep)[w];
					nG1 = (pGradImage + h * nGradWidthStep)[w];
					nG2 = (pGradImage + (h + 1)*nGradWidthStep)[w];
					fEdgeH = h + ((double)(nG0 - nG2) / (nG0 + nG2 - 2 * nG1))*0.5;
					fEdgeW = w;
					//cout<<"h:"<<h<<"  w:"<<w<<"  fw:"<<fEdgeH<<"  "<<nG0<<" "<<nG2<<" "<<nG1<<endl;
					break;
				case 4:
					nG0 = (pGradImage + (h - 1)*nGradWidthStep)[w + 1];
					nG1 = (pGradImage + h * nGradWidthStep)[w];
					nG2 = (pGradImage + (h + 1)*nGradWidthStep)[w - 1];
					fEdgeH = h + ((double)(nG0 - nG2) / (nG0 + nG2 - 2 * nG1))*0.5;
					fEdgeW = w - ((double)(nG0 - nG2) / (nG0 + nG2 - 2 * nG1))*0.5;
					//cout<<"h:"<<h<<"  w:"<<w<<"  fh:"<<fEdgeH<<"  fw:"<<fEdgeW<<"   "<<nG0<<" "<<nG2<<" "<<nG1<<endl;
					break;
				default:
					break;
				}
			}
			(SubEdgeMatH->data.db + h * (SubEdgeMatH->step / sizeof(double)))[w] = fEdgeH;
			(SubEdgeMatW->data.db + h * (SubEdgeMatW->step / sizeof(double)))[w] = fEdgeW;
		}
	}
}