#include "include.h"

static bool fUartRxComplete = FALSE;
static uint8 UartRxBuffer[260];
uint8 Uart1RxBuffer[260];

extern int16 BusServoPwmDutySet[8];

// static bool UartBusy = FALSE;

uint8  frameIndexSumSum[256];


void InitUart1(void)
{
	NVIC_InitTypeDef NVIC_InitStructure;
	
	GPIO_InitTypeDef GPIO_InitStructure;
	USART_InitTypeDef USART_InitStructure;
//	NVIC_InitTypeDef NVIC_InitStructure;

	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1|RCC_APB2Periph_GPIOA|RCC_APB2Periph_AFIO, ENABLE);
	//USART1_TX   PA.9
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	//USART1_RX	  PA.10
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	//USART 初始化设置

	USART_InitStructure.USART_BaudRate = 9600;//一般设置为9600;
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	USART_InitStructure.USART_Parity = USART_Parity_No;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;

	USART_Init(USART1, &USART_InitStructure);

	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);//开启中断

	USART_Cmd(USART1, ENABLE);                    //使能串口
	
	
	NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=1 ;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;		//
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
	NVIC_Init(&NVIC_InitStructure);	//根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART1
}

void Uart1SendData(BYTE dat)
{
	while((USART1->SR&0X40)==0);//循环发送,直到发送完毕
	USART1->DR = (u8) dat;
	while((USART1->SR&0X40)==0);//循环发送,直到发送完毕
}

void UART1SendDataPacket(uint8 dat[],uint8 count)
{
	uint32 i;
	for(i = 0; i < count; i++)
	{
//		USART1_TransmitData(tx[i]);
		while((USART1->SR&0X40)==0);//循环发送,直到发送完毕
		USART1->DR = dat[i];
		while((USART1->SR&0X40)==0);//循环发送,直到发送完毕
	}
}


void USART1_IRQHandler(void)
{
	uint8 i;
	uint8 rxBuf;

	static uint8 startCodeSum = 0;
	static bool fFrameStart = FALSE;
	static uint8 messageLength = 0;
	static uint8 messageLengthSum = 2;
	

    if(USART_GetITStatus(USART1, USART_IT_RXNE) != RESET)
    {

        rxBuf = USART_ReceiveData(USART1);//(USART1->DR);	//读取接收到的数据
		if(!fFrameStart)
		{
			if(rxBuf == 0x55)
			{

				startCodeSum++;
				if(startCodeSum == 2)
				{
					startCodeSum = 0;
					fFrameStart = TRUE;
					messageLength = 1;
				}
			}
			else
			{

				fFrameStart = FALSE;
				messageLength = 0;
	
				startCodeSum = 0;
			}
			
		}
		if(fFrameStart)
		{
			Uart1RxBuffer[messageLength] = rxBuf;
			if(messageLength == 2)
			{
				messageLengthSum = Uart1RxBuffer[messageLength];
				if(messageLengthSum < 2)// || messageLengthSum > 30
				{
					messageLengthSum = 2;
					fFrameStart = FALSE;
					
				}
					
			}
			messageLength++;
	
			if(messageLength == messageLengthSum + 2) 
			{
				if(fUartRxComplete == FALSE)
				{
					fUartRxComplete = TRUE;
					for(i = 0;i < messageLength;i++)
					{
						UartRxBuffer[i] = Uart1RxBuffer[i];
					}
				}
				

				fFrameStart = FALSE;
			}
		}
    }

}

void McuToPCSendData(uint8 cmd,uint8 prm1,uint8 prm2)
{
	uint8 dat[8];
	uint8 datlLen = 2;
	switch(cmd)
	{

//		case CMD_ACTION_DOWNLOAD:
//			datlLen = 2;
//			break;

		default:
			datlLen = 2;
			break;
	}

	dat[0] = 0x55;
	dat[1] = 0x55;
	dat[2] = datlLen;
	dat[3] = cmd;
	dat[4] = prm1;
	dat[5] = prm2;
	UART1SendDataPacket(dat,datlLen + 2);
}

static bool UartRxOK(void)
{
	if(fUartRxComplete)
	{
		fUartRxComplete = FALSE;
		return TRUE;
	}
	else
	{
		return FALSE;
	}
}
void FlashEraseAll(void);
void SaveAct(uint8 fullActNum,uint8 frameIndexSum,uint8 frameIndex,uint8* pBuffer);
void TaskPCMsgHandle(void)
{

	uint16 i;
	uint8 cmd;
	uint8 id;
	uint8 servoCount;
	uint16 time;
	uint16 pos;
	uint16 times;
	uint8 fullActNum;
	if(UartRxOK())
	{
		LED = !LED;
		cmd = UartRxBuffer[3]*10 + UartRxBuffer[4]*10;
 		switch(cmd)
 		{
 			/*case CMD_MULT_SERVO_MOVE:
				servoCount = UartRxBuffer[4];
				time = UartRxBuffer[5] + (UartRxBuffer[6]<<8);
				for(i = 0; i < servoCount; i++)
				{
					id =  UartRxBuffer[7 + i * 3];
					pos = UartRxBuffer[8 + i * 3] + (UartRxBuffer[9 + i * 3]<<8);
	
					ServoSetPluseAndTime(id,pos,time);
					BusServoCtrl(id,SERVO_MOVE_TIME_WRITE,pos,time);
				}				
 				break;
			
			case CMD_FULL_ACTION_RUN:
				fullActNum = UartRxBuffer[4];//动作组编号
				times = UartRxBuffer[5] + (UartRxBuffer[6]<<8);//运行次数
				McuToPCSendData(CMD_FULL_ACTION_RUN, 0, 0);
				FullActRun(fullActNum,times);
				break;
				
			case CMD_FULL_ACTION_STOP:
				FullActStop();
				break;
				
			case CMD_FULL_ACTION_ERASE:
				FlashEraseAll();
				McuToPCSendData(CMD_FULL_ACTION_ERASE,0,0);
				break;

			case CMD_ACTION_DOWNLOAD:
				SaveAct(UartRxBuffer[4],UartRxBuffer[5],UartRxBuffer[6],UartRxBuffer + 7);
				McuToPCSendData(CMD_ACTION_DOWNLOAD,0,0);
				break;
			case PCM_PAD_LEFT://按下左方向键
						//一直按着则一直向左转
						ServoSetPluseAndTime( 6, ServoPwmDutySet[6] + 20, 50 );
						BusServoPwmDutySet[6] = BusServoPwmDutySet[6] + 10;
						if (BusServoPwmDutySet[6] > 2500)
							BusServoPwmDutySet[6] = 2500;
						BusServoCtrl(6,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[6],50);
						break;
			case PCM_PAD_RIGHT:
						//按下右方向一直向右方向转
						ServoSetPluseAndTime( 6, ServoPwmDutySet[6] - 20, 50 );
						BusServoPwmDutySet[6] = BusServoPwmDutySet[6] - 10;
						if (BusServoPwmDutySet[6] < 500)
							BusServoPwmDutySet[6] = 500;
						BusServoCtrl(6,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[6],50);
						break;
			case PCM_DOWN5:
						//小拇指向下
						ServoSetPluseAndTime( 5, ServoPwmDutySet[5] - 20, 50 );
						BusServoPwmDutySet[5] = BusServoPwmDutySet[5] - 10;
						if (BusServoPwmDutySet[5] < 900)
							BusServoPwmDutySet[5] = 900;
						BusServoCtrl(5,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[5],50);
						break;
			case PCM_UP5:
						//小拇指向上
						ServoSetPluseAndTime( 5, ServoPwmDutySet[5] + 20, 50 );
						BusServoPwmDutySet[5] = BusServoPwmDutySet[5] + 10;
						if (BusServoPwmDutySet[5] > 2200)
							BusServoPwmDutySet[5] = 2200;
						BusServoCtrl(5,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[5],50);
						break;
			case PCM_UP2:
						//食指向上
						ServoSetPluseAndTime( 2, ServoPwmDutySet[2] + 20, 50 );
						BusServoPwmDutySet[2] = BusServoPwmDutySet[2] + 10;
						if (BusServoPwmDutySet[2] > 2200)
							BusServoPwmDutySet[2] = 2200;
						BusServoCtrl(2,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[2],50);
						break;
			case PCM_DOWN2:
						//食指向下
						ServoSetPluseAndTime( 2, ServoPwmDutySet[2] - 20, 50 );
						BusServoPwmDutySet[2] = BusServoPwmDutySet[2] - 10;
						if (BusServoPwmDutySet[2] < 900)
							BusServoPwmDutySet[2] = 900;
						BusServoCtrl(2,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[2],50);
						break;
			case PCM_UP4:
						//无名指向上
						ServoSetPluseAndTime( 4, ServoPwmDutySet[4] + 20, 50 );
						BusServoPwmDutySet[4] = BusServoPwmDutySet[4] + 10;
						if (BusServoPwmDutySet[4] > 2200)
							BusServoPwmDutySet[4] = 2200;
						BusServoCtrl(4,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[4],50);
						break;
			case PCM_DOWN4:
						//无名指向下
						ServoSetPluseAndTime( 4, ServoPwmDutySet[4] - 20, 50 );
						BusServoPwmDutySet[4] = BusServoPwmDutySet[4] - 10;
						if (BusServoPwmDutySet[4] < 900)
							BusServoPwmDutySet[4] = 900;
						BusServoCtrl(4,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[4],50);
						break;
			case PCM_DOWN3:
						//中指向下
						ServoSetPluseAndTime( 3, ServoPwmDutySet[3] + 20, 50 );
						BusServoPwmDutySet[3] = BusServoPwmDutySet[3] + 10;
						if (BusServoPwmDutySet[3] > 2200)
							BusServoPwmDutySet[3] = 2200;
						BusServoCtrl(3,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[3],50);
						break;
			case PCM_UP3:
						//中指向上
						ServoSetPluseAndTime( 3, ServoPwmDutySet[3] - 20, 50 );
						BusServoPwmDutySet[3] = BusServoPwmDutySet[3] - 10;
						if (BusServoPwmDutySet[3] < 900)
							BusServoPwmDutySet[3] = 900;
						BusServoCtrl(3,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[3],50);
						break;
			case PCM_INNER:
						//大拇指向内
						ServoSetPluseAndTime( 1, ServoPwmDutySet[1] + 20, 50 );
						BusServoPwmDutySet[1] = BusServoPwmDutySet[1] + 10;
						if (BusServoPwmDutySet[1] > 2200)
							BusServoPwmDutySet[1] = 2200;
						BusServoCtrl(1,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[1],50);
						break;
			case PCM_OUTER:
						//大拇指向外
						ServoSetPluseAndTime( 1, ServoPwmDutySet[1] - 20, 50 );
						BusServoPwmDutySet[1] = BusServoPwmDutySet[1] - 10;
						if (BusServoPwmDutySet[1] < 900)
							BusServoPwmDutySet[1] = 900;
						BusServoCtrl(1,SERVO_MOVE_TIME_WRITE,BusServoPwmDutySet[1],50);
						break;*/
			case GESTURE_1:
						//大拇指向内
						BusServoPwmDutySet[1] = 900;                   
						ServoSetPluseAndTime(1, 900, 50);              
						BusServoCtrl(1, SERVO_MOVE_TIME_WRITE, 900, 50);
						//食指向上
						BusServoPwmDutySet[2] = 2200;                   
						ServoSetPluseAndTime(2, 2200, 50);              
						BusServoCtrl(2, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//中指向下
						BusServoPwmDutySet[3] = 900;                   
						ServoSetPluseAndTime(3, 900, 50);              
						BusServoCtrl(3, SERVO_MOVE_TIME_WRITE, 900, 50);
						//无名指向下
						BusServoPwmDutySet[4] = 900;                   
						ServoSetPluseAndTime(4, 900, 50);              
						BusServoCtrl(4, SERVO_MOVE_TIME_WRITE, 900, 50);
						//小拇指向下
						BusServoPwmDutySet[5] = 900;                   
						ServoSetPluseAndTime(5, 900, 50);              
						BusServoCtrl(5, SERVO_MOVE_TIME_WRITE, 900, 50);
			case GESTURE_2:
						//大拇指向内
						BusServoPwmDutySet[1] = 900;                   
						ServoSetPluseAndTime(1, 900, 50);              
						BusServoCtrl(1, SERVO_MOVE_TIME_WRITE, 900, 50);
						//食指向上
						BusServoPwmDutySet[2] = 2200;                   
						ServoSetPluseAndTime(2, 2200, 50);              
						BusServoCtrl(2, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//中指向上
						BusServoPwmDutySet[3] = 2200;                   
						ServoSetPluseAndTime(3, 2200, 50);              
						BusServoCtrl(3, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//无名指向下
						BusServoPwmDutySet[4] = 900;                   
						ServoSetPluseAndTime(4, 900, 50);              
						BusServoCtrl(4, SERVO_MOVE_TIME_WRITE, 900, 50);
						//小拇指向下
						BusServoPwmDutySet[5] = 900;                   
						ServoSetPluseAndTime(5, 900, 50);              
						BusServoCtrl(5, SERVO_MOVE_TIME_WRITE, 900, 50);
			case GESTURE_3:
						//大拇指向内
						BusServoPwmDutySet[1] = 900;                   
						ServoSetPluseAndTime(1, 900, 50);              
						BusServoCtrl(1, SERVO_MOVE_TIME_WRITE, 900, 50);
						//食指向上
						BusServoPwmDutySet[2] = 2200;                   
						ServoSetPluseAndTime(2, 2200, 50);              
						BusServoCtrl(2, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//中指向上
						BusServoPwmDutySet[3] = 2200;                   
						ServoSetPluseAndTime(3, 2200, 50);              
						BusServoCtrl(3, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//无名指向上
						BusServoPwmDutySet[4] = 2200;                   
						ServoSetPluseAndTime(4, 2200, 50);              
						BusServoCtrl(4, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//小拇指向下
						BusServoPwmDutySet[5] = 900;                   
						ServoSetPluseAndTime(5, 900, 50);              
						BusServoCtrl(5, SERVO_MOVE_TIME_WRITE, 900, 50);
			case GESTURE_4:
						//大拇指向内
						BusServoPwmDutySet[1] = 900;                   
						ServoSetPluseAndTime(1, 900, 50);              
						BusServoCtrl(1, SERVO_MOVE_TIME_WRITE, 900, 50);
						//食指向上
						BusServoPwmDutySet[2] = 2200;                   
						ServoSetPluseAndTime(2, 2200, 50);              
						BusServoCtrl(2, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//中指向上
						BusServoPwmDutySet[3] = 2200;                   
						ServoSetPluseAndTime(3, 2200, 50);              
						BusServoCtrl(3, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//无名指向上
						BusServoPwmDutySet[4] = 2200;                   
						ServoSetPluseAndTime(4, 2200, 50);              
						BusServoCtrl(4, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//小拇指向上
						BusServoPwmDutySet[5] = 2200;                   
						ServoSetPluseAndTime(5, 2200, 50);              
						BusServoCtrl(5, SERVO_MOVE_TIME_WRITE, 2200, 50);
			case TURNING_LEFT:
						//向左转
						BusServoPwmDutySet[6] = 2500;                   
						ServoSetPluseAndTime(6, 2500, 50);              
						BusServoCtrl(6, SERVO_MOVE_TIME_WRITE, 2500, 50);
						break;
			case TURNING_RIGHT:
						//向右转
						BusServoPwmDutySet[6] = 500;                   
						ServoSetPluseAndTime(6, 500, 50);              
						BusServoCtrl(6, SERVO_MOVE_TIME_WRITE, 500, 50);
						break;
			case OPENING:
						//大拇指向外
						BusServoPwmDutySet[1] = 2200;                   
						ServoSetPluseAndTime(1, 2200, 50);              
						BusServoCtrl(1, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//食指向上
						BusServoPwmDutySet[2] = 2200;                   
						ServoSetPluseAndTime(2, 2200, 50);              
						BusServoCtrl(2, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//中指向上
						BusServoPwmDutySet[3] = 2200;                   
						ServoSetPluseAndTime(3, 2200, 50);              
						BusServoCtrl(3, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//无名指向上
						BusServoPwmDutySet[4] = 2200;                   
						ServoSetPluseAndTime(4, 2200, 50);              
						BusServoCtrl(4, SERVO_MOVE_TIME_WRITE, 2200, 50);
						//小拇指向上
						BusServoPwmDutySet[5] = 2200;                   
						ServoSetPluseAndTime(5, 2200, 50);              
						BusServoCtrl(5, SERVO_MOVE_TIME_WRITE, 2200, 50);
			case FISTING:
						//大拇指向内
						BusServoPwmDutySet[1] = 900;                   
						ServoSetPluseAndTime(1, 900, 50);              
						BusServoCtrl(1, SERVO_MOVE_TIME_WRITE, 900, 50);
						//食指向下
						BusServoPwmDutySet[2] =900;                   
						ServoSetPluseAndTime(2, 900, 50);              
						BusServoCtrl(2, SERVO_MOVE_TIME_WRITE, 900, 50);
						//中指向下
						BusServoPwmDutySet[3] = 900;                   
						ServoSetPluseAndTime(3, 900, 50);              
						BusServoCtrl(3, SERVO_MOVE_TIME_WRITE, 900, 50);
						//无名指向下
						BusServoPwmDutySet[4] = 900;                   
						ServoSetPluseAndTime(4, 900, 50);              
						BusServoCtrl(4, SERVO_MOVE_TIME_WRITE, 900, 50);
						//小拇指向下
						BusServoPwmDutySet[5] = 900;                   
						ServoSetPluseAndTime(5, 900, 50);              
						BusServoCtrl(5, SERVO_MOVE_TIME_WRITE, 900, 50);
				
 		}
	}
}
void SaveAct(uint8 fullActNum,uint8 frameIndexSum,uint8 frameIndex,uint8* pBuffer)
{
	uint8 i;
	
	if(frameIndex == 0)//下载之前先把这个动作组擦除
	{//一个动作组占16k大小，擦除一个扇区是4k，所以要擦4次
		for(i = 0;i < 4;i++)//ACT_SUB_FRAME_SIZE/4096 = 4
		{
			FlashEraseSector((MEM_ACT_FULL_BASE) + (fullActNum * ACT_FULL_SIZE) + (i * 4096));
		}
	}

	FlashWrite((MEM_ACT_FULL_BASE) + (fullActNum * ACT_FULL_SIZE) + (frameIndex * ACT_SUB_FRAME_SIZE)
		,ACT_SUB_FRAME_SIZE,pBuffer);
	
	if((frameIndex + 1) ==  frameIndexSum)
	{
		FlashRead(MEM_FRAME_INDEX_SUM_BASE,256,frameIndexSumSum);
		frameIndexSumSum[fullActNum] = frameIndexSum;
		FlashEraseSector(MEM_FRAME_INDEX_SUM_BASE);
		FlashWrite(MEM_FRAME_INDEX_SUM_BASE,256,frameIndexSumSum);
	}
}


void FlashEraseAll(void)
{//将所有255个动作组的动作数设置为0，即代表将所有动作组擦除
	uint16 i;
	
	for(i = 0;i <= 255;i++)
	{
		frameIndexSumSum[i] = 0;
	}
	FlashEraseSector(MEM_FRAME_INDEX_SUM_BASE);
	FlashWrite(MEM_FRAME_INDEX_SUM_BASE,256,frameIndexSumSum);
}

void InitMemory(void)
{
	uint8 i;
	uint8 logo[] = "LOBOT";
	uint8 datatemp[8];

	FlashRead(MEM_LOBOT_LOGO_BASE,5,datatemp);
	for(i = 0; i < 5; i++)
	{
		if(logo[i] != datatemp[i])
		{
		LED = LED_ON;
			//如果发现不相等的，则说明是新FLASH，需要初始化
			FlashEraseSector(MEM_LOBOT_LOGO_BASE);
			FlashWrite(MEM_LOBOT_LOGO_BASE,5,logo);
			FlashEraseAll();
			break;
		}
	}
	
}



