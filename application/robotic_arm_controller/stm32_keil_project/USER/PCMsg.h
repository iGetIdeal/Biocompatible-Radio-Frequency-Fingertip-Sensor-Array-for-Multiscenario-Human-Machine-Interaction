#ifndef _PC_MSG_H_
#define _PC_MSG_H_
#include "include.h"



//指令
#define CMD_MULT_SERVO_MOVE					13	//多个舵机相同时间运动
#define CMD_FULL_ACTION_RUN					16
#define CMD_FULL_ACTION_STOP				17
#define CMD_FULL_ACTION_ERASE				18
#define CMD_ACTION_DOWNLOAD					25
/*#define PCM_PAD_LEFT           1    //向左转
#define PCM_PAD_RIGHT          2    //向右转
#define PCM_UP5                3    //小拇指向上
#define PCM_DOWN5              4    //小拇指向下
#define PCM_UP2                5    //食指向上
#define PCM_DOWN2              6		//食指向下
#define PCM_UP4                7		//无名指向上
#define PCM_DOWN4              8		//无名指向下
#define PCM_DOWN3              9		//中指向下
#define PCM_UP3                10 	//中指向上
#define PCM_INNER              11		//大拇指向内
#define PCM_OUTER              12		//大拇指向外*/
#define GESTURE_1							 1
#define GESTURE_2							 2
#define GESTURE_3							 3
#define GESTURE_4							 4
#define TURNING_LEFT					 5
#define TURNING_RIGHT					 6
#define OPENING							   7
#define FISTING						     8




//存放起始地址
#define MEM_LOBOT_LOGO_BASE					0L	//"LOBOT"存放基地址，用于识别是否是新FLASH
#define MEM_FRAME_INDEX_SUM_BASE			4096L//每个动作组有多少动作，从这个地址开始存放，共计256个动作组
#define MEM_ACT_FULL_BASE					8192L//动作组文件从这个地址开始存放

//大小
#define ACT_SUB_FRAME_SIZE					64L		//一个动作帧占64字节空间
#define ACT_FULL_SIZE						16384L	//16KB,一套完整动作组占14kb字节




static bool UartRxOK(void);
void FlashEraseAll(void);
void McuToPCSendData(uint8 cmd,uint8 prm1,uint8 prm2);
void InitUart1(void);
void SaveAct(uint8 fullActNum,uint8 frameIndexSum,uint8 frameIndex,uint8* pBuffer);
void TaskPCMsgHandle(void);
void InitMemory(void);

#endif

