#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<ctype.h>
#include<cstdlib>

int constant;
FILE *fp;//打开文件

typedef struct mold//定义数据类型
{
	int num;
	int year;
	int month;
	int day;
    char type[25];
	char fname[25];
	double amount;
}SJLX;

SJLX pro[200];


void menu();//主菜单函数，功能：数据处理、数据查询、数据统计、退出
void datainquiry();//数据查询函数，功能：查询所有数据、按年月日查询、按名称查询、返回
void addup();//数据统计，功能：课按年、月、日分别统计收入、支出、结余金额
void yeard();//按年统计函数
void monthd();//按月统计函数
void dayd();//按日统计函数
void allinquiry();//查询所有信息
void yeara();//按年查询函数
void montha();//按月查询函数
void daya();//按日查询函数
void namea();//名称查询
void incomea();//按收支查询函数
void sorting();//排序函数，功能：可按日期、名称、金额排序
void numbers();//序号排序
void dates();//日期排序
void names();//名称排序
void amounts();//金额排序
void inputdata();//输入信息函数
void revisedata();//修改信息函数
void deletedata();//删除信息函数
int readfile();//读取帐务信息
void writefile(int x);//记录帐务信息
char inputcheck();//判断输入是否为数字

void menu()//主菜单函数，功能：数据处理、数据查询、数据统计、退出
{
	char input;
	printf("                 欢迎来到个人财务系统：\n");
	printf("                     1.数据录入\n");
	printf("                     2.数据查询\n");
	printf("                     3.数据修改\n");
	printf("                     4.数据删除\n");
	printf("                     5.数据统计\n");
    printf("                     6.数据排序\n");
	printf("                     0.退出\n");
	printf("\t****************************************\n");
	printf("请输入选择:");
	input = inputcheck();
	switch(input)
	{
		case '1': inputdata();break;
		case '2': datainquiry();break;
		case '3': revisedata();break;
		case '4': deletedata();break;
		case '5': addup();break;
		case '6': sorting();break;
		case '0': exit(0);break;
	}
}
int readfile()//读取账务信息
{
	int k = 0;
	if((fp=fopen("zhangmu.txt","r"))==NULL)
	{
		printf("初始化时未发现文件，已建立。\n");
		if((fp=fopen("zhangmu.txt","w"))==NULL)
		{
			printf("初始化失败!将无法进行存储信息。请重新登录系统。\n");
			exit(0);
		}
	}
	do
	{
		fread(&pro[k],sizeof(SJLX),1,fp);
		k++;
	}
	while(pro[k-1].num!=0);
	return k-1;
	fclose(fp);
}

void writefile(int x)//记录账务信息
{
	int i = 0;
	if((fp=fopen("zhangmu.txt","w"))==NULL)
	{
		printf("写文件失败！\n");exit(0);
	}
	if(x==1)
	{
		for(i=0;i<constant;i++)
		{
			fwrite(&pro[i],sizeof(SJLX),1,fp);
		}
		printf("文件修改成功。\n");
		fclose(fp);
	}
	else;
}
void inputdata()//输入信息函数
{
	int i,word=1,x,k=0;
	for(;constant<200 && word==1;constant++)
	{
		printf("请输入编号：\n");
		scanf("%d",&pro[constant].num);
		printf("请输入年：\n");
		scanf("%d",&pro[constant].year);
		printf("请输入月：\n");
		scanf("%d",&pro[constant].month);
		printf("请输入日：\n");
		scanf("%d",&pro[constant].day);
		printf("请输入收支类型：\n");
		scanf("%s",pro[constant].type);
		printf("请输入收支项目名称：\n");
		scanf("%s",pro[constant].fname);
		printf("请输入收支金额：\n");
		scanf("%lf",&pro[constant].amount);
		printf("要继续吗？\n1.是  2.退出\n");
		scanf("%d",&word);
	}
	for(i = 0;i < constant;i++)
	{
		printf("%-5d%-5d%-5d%-5d%-10s%-10s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
	}
	printf("\n新建的账目需要存档吗?\t1.是    2.否\n");
	scanf("%d",&x);
	writefile(x);
	fflush(stdin);	
    menu();
}
void datainquiry()//数据查询函数，功能：查询所有数据、按年月日查询、按收入和支出查询、返回
{
	char input;
	system("cls");
	printf("    数据查询：\n");
	printf("  1.所有数据\n");
	printf("  2.按年查询\n");
	printf("  3.按月查询\n");
	printf("  4.按日查询\n");
	printf("  5.按收支查询\n");
	printf("  6.按名称查询\n");
	printf("  0.返回\n");
	printf("请输入选择:");
	input = inputcheck();
	switch(input)
	{
		case '1': allinquiry();break;
		case '2': yeara();break;
		case '3': montha();break;
		case '4': daya();break;
		case '5': incomea();break;
        case '6': namea();break;
		case '0': menu();break;
	}fflush(stdin);
	menu();
}
void allinquiry()//所有信息查询
{
	int i;
	printf("编号      年      月    日  收支类型\t收支项名称          金额\n");
	for(i = 0;i < constant;i++)
	{
		printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
	}
}
void yeara()//按年查询
{
	int y=0,m=0,i=0;
	printf("请输入年份");
	scanf("%d",&y);
	printf("编号      年      月    日  收支类型\t收支项名称          金额\n");
	for(i = 0;i < constant;i++)
	{
		if(pro[i].year == y)
		{
			printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
		}
	}
}

void montha()//按月查询
{
	int y,m,i;
	printf("请输入年份");
	scanf("%d",&y);
	printf("请输入月份");
	scanf("%d",&m);
	printf("编号      年      月    日  收支类型\t收支项名称          金额\n");
	for(i = 0;i < constant;i++)
	{
		if(pro[i].year == y && pro[i].month == m)
		{
			printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
		}
	}
}
void daya()//按日查询
{
	int y,m,n,i;
	printf("请输入年份");
	scanf("%d",&y);
	printf("请输入月份");
	scanf("%d",&m);
	printf("请输入日");
	scanf("%d",&n);
	printf("编号      年      月    日  收支类型\t收支项名称          金额\n");
	for(i=0;i<constant;i++)
	{
		if(pro[i].year == y && pro[i].month == m&&pro[i].day==n)
		{
			printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
		}
	}
}
void incomea()//按收支查询
{
	char ch;
	int i;
	printf("\t\t\t1.收入\n\t\t\t2.支出\n请选择:");
	ch = inputcheck();
	switch(ch)
	{
		case '1':
			{
				printf("编号      年      月    日  收支项名称          金额\n");
				for(i = 0;i < constant;i++)
				{
					if(!(strcmp(pro[i].type,"收入")))
					{
						printf("%-10d%-9d%-5d%-5d%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].fname,pro[i].amount);
					}
				}
			};break;
		case '2':
			{
				printf("编号      年      月    日  收支项名称          金额\n");
				for(i = 0;i < constant;i++)
				{
					if(!(strcmp(pro[i].type,"支出")))
					{
						printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
					}
				}
			};break;
	}
}
void namea()//按名称查询
{   int i;
	char k[50];
	printf("请输入待查询的收支项目名称\n");
	scanf("%s",&k);
	for(i = 0;i < constant;i++)
	{
	   if(!(strcmp(pro[i].fname,k)))
       printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
	}
}

void revisedata()//修改信息函数
{
	int ch,i;
	char input;
	system("cls");
	printf("请查询要修改信息：\n");
	printf("1.所有数据\n");
	printf("2.按年查询\n");
	printf("3.按月查询\n");
	printf("4.按日查询\n");
	printf("5.按收支查询\n");
	printf("6.按名称查询\n");
	printf("请输入选择:");
	input = inputcheck();
	switch(input)
	{
		case '1': allinquiry();break;
		case '2': yeara();break;
		case '3': montha();break;
		case '4': daya();break;
		case '5': incomea();break;
        case '6': namea();break;
	}
	printf("请输入要修改的收支编号：\n");
	scanf("%d",&ch);
	for(i = 0;i < constant;i ++)
	{
		if(pro[i].num == ch)
		{
			printf("请输入修改信息：\n");
			printf("请输入编号：\n");
			scanf("%d",&pro[i].num);
			printf("请输入年：\n");
			scanf("%d",&pro[i].year);
			printf("请输入月：\n");
			scanf("%d",&pro[i].month);
			printf("请输入日：\n");
			scanf("%d",&pro[i].day);
			printf("请输入收支类型：\n");
		    scanf("%s",pro[constant].type);
			printf("请输入收支项目名称：\n");
			scanf("%s",pro[i].fname);
			printf("请输入收支金额：\n");
			scanf("%lf",&pro[i].amount);
			writefile(1);
			fflush(stdin);
			menu();
		}
	}
}

void deletedata()//删除信息函数
{
	int ch,i;
	char input;
	system("cls");
	printf("请查询要删除信息：\n");
	printf("1.所有数据\n");
	printf("2.按年查询\n");
	printf("3.按月查询\n");
	printf("4.按日查询\n");
	printf("5.按收支查询\n");
	printf("6.按名称查询\n");
	printf("请输入选择:");
	input = inputcheck();
	switch(input)
	{
		case '1': allinquiry();break;
		case '2': yeara();break;
		case '3': montha();;break;
		case '4': daya();break;
		case '5': incomea();break;
        case '6': namea();break;
	}
    printf("请输入要删除的收支编号：\n");
	scanf("%d",&ch);
	for(i = 0;i < constant + 1;i ++)
	{
		if(pro[i].num == ch)
		{
			pro[i].num = pro[i+1].num;
			pro[i].year = pro[i+1].year;
			pro[i].month = pro[i+1].month;
			pro[i].day = pro[i+1].day;
			strcpy(pro[i].fname,pro[i+1].fname);
			pro[i].amount = pro[i+1].amount;
			constant = constant - 1;
			writefile(1);
			fflush(stdin);
			menu();
		}
	}
}

void addup()//数据统计函数
{   
	char input;
	system("cls");
    printf("数据统计：\n");
    printf("1.按年统计\n");
	printf("2.按月统计\n");
    printf("3.按日统计\n");
    printf("0.返回\n");
	printf("#请选择#\n");
    input = inputcheck();
	switch(input)
	{
		case '1': yeard();break;
		case '2': monthd();break;
		case '3': dayd();break;
		case '0': menu();break;
	}fflush(stdin);
	menu();
}
void yeard()//按年统计
{
	char input;
	int y;
    int i;
    double sum=0;
	system("cls");
	printf("\t\t********年度统计*********\n\n");
	printf("\t\t1.收入统计\n");
	printf("\t\t2.支出统计\n");
	printf("\t\t3.结余金额\n");
	printf("\t\t0.返回\n");
	printf("请选择统计类型:");
	input = inputcheck();
    printf("请输入年份");
	scanf("%d",&y);
	switch(input)
	{
		case '1':
			{
	            for(i=0;i < constant;i++)
				{
				   if(pro[i].year==y&&!(strcmp(pro[i].type,"收入")))
						sum+=pro[i].amount;
				}
				printf("本年度总收入金额：%lf\n",sum);
			};break;

					
		case '2': {
	            for(i = 0;i < constant;i++)
				{
				   if(pro[i].year==y&&!(strcmp(pro[i].type,"支出")))
					{
						sum+=pro[i].amount;
					}
				}
				printf("本年度总支出金额：%lf\n",sum);
			};break;
        case'3':{
	            for(i = 0;i < constant;i++)
				{
					if(pro[i].year==y)
						sum+=pro[i].amount;
				}
				printf("本年度总结余金额：%lf\n",sum);
				};break;
		case '0': menu();break;
	}
}
void monthd()//按月统计
{
	char input;
	int y,m;
    int i;
    double sum=0;
	system("cls");
	printf("\t\t*********月度统计********\n\n");
	printf("\t\t1.收入统计\n");
	printf("\t\t2.支出统计\n");
	printf("\t\t3.结余金额\n");
	printf("\t\t0.返回\n");
	printf("请选择统计类型:");
	input = inputcheck();
	printf("请输入年份");
	scanf("%d",&y);
	printf("请输入月份");
	scanf("%d",&m);
	switch(input)
	{
		case '1':
			{
			
	            for(i = 0;i < constant;i++)
				{
				   if(pro[i].year==y&&pro[i].month==m&&!(strcmp(pro[i].type,"收入")))
					{
						sum+=pro[i].amount;
					}
				}
				printf("本月度总收入金额：%lf\n",sum);
			};break;

					
		case '2': {
					
	            for(i = 0;i < constant;i++)
				{
				   if(pro[i].year==y&&pro[i].month==m&&!(strcmp(pro[i].type,"支出")))
					{
						sum+=pro[i].amount;
					}
				}
				printf("本月度总支出金额：%lf\n",sum);
			};break;
        case'3':{
			
	            for(i = 0;i < constant;i++)
				{
					if(pro[i].year==y&&pro[i].month==m)
						sum+=pro[i].amount;
				}
				printf("本月度总结余金额：%lf\n",sum);
				};break;
		case '0': menu();break;
	}
}
void dayd()//按日统计
{
	char input;
	int y,m,n;
    int i;
    double sum=0;
	system("cls");
	printf("\t\t*********每日统计*********\n\n");
	printf("\t1.收入统计\n");
	printf("\t2.支出统计\n");
	printf("\t3.结余金额\n");
	printf("\t0.返回\n");
	printf("请选择统计类型:");
	printf("请输入年份");
	scanf("%d",&y);
	printf("请输入月份");
	scanf("%d",&m);
	printf("请输入日");
	scanf("%d",&n);
	input = inputcheck();
	switch(input)
	{
		case '1':
			{
				
	            for(i = 0;i < constant;i++)
				{
				   if(pro[i].year==y&&pro[i].month==m&&pro[i].day==n&&!(strcmp(pro[i].type,"收入")))
					{
						sum+=pro[i].amount;
					}
				}
				printf("该日总收入金额：%lf\n",sum);
			};break;

					
		case '2': {
				
	            for(i = 0;i < constant;i++)
				{
				   if(pro[i].year==y&&pro[i].month==m&&pro[i].day==n&&!(strcmp(pro[i].type,"支出")))
					{
						sum+=pro[i].amount;
					}
				}
				printf("该日总支出金额：%lf\n",sum);
			};break;
        case'3':{
			
	            for(i = 0;i < constant;i++)
				{
					if(pro[i].year==y&&pro[i].month==m&&pro[i].day==n)
						sum+=pro[i].amount;
				}
				printf("本日总结余金额：%lf\n",sum);
				};break;
		case '0': menu();break;
	}
}



void sorting()//排序函数
{
	char input;
	system("cls");
	printf("数据排序：\n");
    printf("1.按名称\n");
	printf("2.按金额\n");
	printf("3.按日期\n");
	printf("0.返回\n");
	printf("请输入选择:");
	input = inputcheck();
	switch(input)
	{
		
		case '1': names();menu();break;
		case '2': amounts();menu();break;
		case '3': dates();menu();break;
		case '0': menu();break;
	}fflush(stdin);
	menu();
}
void numbers()//按序号排序
{
	int i,j,temp,noswap,n=0;
    for(i = 0;i < constant;i++)
	{
		if(pro[i].num!=NULL)
			n++;
	}
	for(i=0;i<n-1;i++)
	{
		noswap=1;
		for(j=n-1;j>i;j--)
			if(pro[j].num <pro[j-1].num )
			{
				temp=pro[j-1].num;
				pro[j-1].num =pro[j].num ;
				pro[j].num =temp;
				noswap=0;
			}
			if(noswap==1)
			return;
	}
}

void dates()//按日期排序
{
    int i,j,last,n=0;
	SJLX temp;
    for(i = 0;i < constant;i++)
	{
		if(pro[i].day!=NULL)
			n++;
	}
	for(i=0;i<n-1;i++)
	{
		last=n-1;
		for(j=n-1;j>i;j--)
			if(pro[j].year<pro[j-1].year )
			{
				temp=pro[j-1];
				pro[j-1] =pro[j];
				pro[j]=temp;
				last=j;
			}
			i=last;
		
	}
    for(i=0;i<n-1;i++)
	{
		last=n-1;
		for(j=n-1;j>i;j--)
			if(pro[j].year<=pro[j-1].year&&pro[j].month<pro[j-1].month )
			{
				temp=pro[j-1];
				pro[j-1] =pro[j];
				pro[j]=temp;
				last=j;
			}
			i=last;
		
	}   
   for(i=0;i<n-1;i++)
	{
		last=n-1;
		for(j=n-1;j>i;j--)
			if(pro[j].year<=pro[j-1].year&&pro[j].month<=pro[j-1].month&&pro[j].day<pro[j-1].day)
			{
				temp=pro[j-1];
				pro[j-1] =pro[j];
				pro[j]=temp;
				last=j;
			}
			i=last;
			
	}   
   numbers();
   printf("编号      年      月    日  收支类型\t收支项名称          金额\n");
   for(i = 0;i < constant;i++)		
             printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
}
void names()//按名称排序
{
    int i,j,last,n=0;
	SJLX temp;
    for(i = 0;i < constant;i++)
	{
		if(pro[i].fname!=NULL)
			n++;
	}
	for(i=0;i<n-1;i++)
	{
		last=n-1;
		for(j=n-1;j>i;j--)
			if((strcmp(pro[j].fname,pro[j-1].fname))<0)
			{
				temp=pro[j-1];
				pro[j-1] =pro[j];
				pro[j]=temp;
				last=j;
			}
			i=last;
	} 
   numbers();
   printf("编号      年      月    日  收支类型\t收支项名称          金额\n");
   for(i = 0;i < constant;i++)		
             printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
}
void amounts()//按金额排序
{
    int i,j,last,n=0;
	SJLX temp;
    for(i = 0;i < constant;i++)
	{
		if(pro[i].fname!=NULL)
			n++;
	}
	for(i=0;i<n-1;i++)
	{
		last=n-1;
		for(j=n-1;j>i;j--)
			if((pro[j].amount<pro[j-1].amount))
			{
				temp=pro[j-1];
				pro[j-1] =pro[j];
				pro[j]=temp;
				last=j;
			}
			i=last;
	} 
   numbers();
   printf("编号      年      月    日  收支类型\t收支项名称          金额\n");
   for(i = 0;i < constant;i++)		
             printf("%-10d%-9d%-5d%-5d%-15s%-15s%-5lf\n",pro[i].num,pro[i].year,pro[i].month,pro[i].day,pro[i].type,pro[i].fname,pro[i].amount);
}


char inputcheck()//判断输入是否为数字
{
	char x;
	if(isdigit(x = getchar()))
	{
		printf("%c\n",x);
		getchar();
		return(x);
	}
	else
	{
		printf("%c\n",x);
		printf("请输入数字!\n");
		printf("请重新选择序号:");
		inputcheck();
	}
	return 0;
}

void main()//主函数
{
	system("color f1");
	constant = readfile();
	menu();
}
