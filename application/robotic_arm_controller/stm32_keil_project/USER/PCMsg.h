#ifndef _PC_MSG_H_
#define _PC_MSG_H_
#include "include.h"



//ָ��
#define CMD_MULT_SERVO_MOVE					13	//��������ͬʱ���˶�
#define CMD_FULL_ACTION_RUN					16
#define CMD_FULL_ACTION_STOP				17
#define CMD_FULL_ACTION_ERASE				18
#define CMD_ACTION_DOWNLOAD					25
/*#define PCM_PAD_LEFT           1    //����ת
#define PCM_PAD_RIGHT          2    //����ת
#define PCM_UP5                3    //СĴָ����
#define PCM_DOWN5              4    //СĴָ����
#define PCM_UP2                5    //ʳָ����
#define PCM_DOWN2              6		//ʳָ����
#define PCM_UP4                7		//����ָ����
#define PCM_DOWN4              8		//����ָ����
#define PCM_DOWN3              9		//��ָ����
#define PCM_UP3                10 	//��ָ����
#define PCM_INNER              11		//��Ĵָ����
#define PCM_OUTER              12		//��Ĵָ����*/
#define GESTURE_1							 1
#define GESTURE_2							 2
#define GESTURE_3							 3
#define GESTURE_4							 4
#define TURNING_LEFT					 5
#define TURNING_RIGHT					 6
#define OPENING							   7
#define FISTING						     8




//�����ʼ��ַ
#define MEM_LOBOT_LOGO_BASE					0L	//"LOBOT"��Ż���ַ������ʶ���Ƿ�����FLASH
#define MEM_FRAME_INDEX_SUM_BASE			4096L//ÿ���������ж��ٶ������������ַ��ʼ��ţ�����256��������
#define MEM_ACT_FULL_BASE					8192L//�������ļ��������ַ��ʼ���

//��С
#define ACT_SUB_FRAME_SIZE					64L		//һ������֡ռ64�ֽڿռ�
#define ACT_FULL_SIZE						16384L	//16KB,һ������������ռ14kb�ֽ�




static bool UartRxOK(void);
void FlashEraseAll(void);
void McuToPCSendData(uint8 cmd,uint8 prm1,uint8 prm2);
void InitUart1(void);
void SaveAct(uint8 fullActNum,uint8 frameIndexSum,uint8 frameIndex,uint8* pBuffer);
void TaskPCMsgHandle(void);
void InitMemory(void);

#endif

