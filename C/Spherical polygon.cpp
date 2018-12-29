// finalDlg.cpp : implementation file
//

#include "stdafx.h"
#include "final.h"
#include "finalDlg.h"
#include "math.h"
#include "stdio.h"
#include <iostream>
#include <string>
using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#define pi 3.141592653589793
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	//{{AFX_DATA(CAboutDlg)
	enum { IDD = IDD_ABOUTBOX };
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAboutDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	//{{AFX_MSG(CAboutDlg)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
	//{{AFX_DATA_INIT(CAboutDlg)
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFinalDlg dialog

CFinalDlg::CFinalDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CFinalDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CFinalDlg)
	m_count = _T("");
	m_x1 = 0.0;
	m_x2 = 0.0;
	m_x3 = 0.0;
	m_x4 = 0.0;
	m_y1 = 0.0;
	m_y2 = 0.0;
	m_y3 = 0.0;
	m_y4 = 0.0;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CFinalDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CFinalDlg)
	DDX_Text(pDX, IDC_EDIT11, m_count);
	DDX_Text(pDX, IDC_EDIT1, m_x1);
	DDX_Text(pDX, IDC_EDIT2, m_x2);
	DDX_Text(pDX, IDC_EDIT3, m_x3);
	DDX_Text(pDX, IDC_EDIT4, m_x4);
	DDX_Text(pDX, IDC_EDIT5, m_y1);
	DDX_Text(pDX, IDC_EDIT6, m_y2);
	DDX_Text(pDX, IDC_EDIT7, m_y3);
	DDX_Text(pDX, IDC_EDIT8, m_y4);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CFinalDlg, CDialog)
	//{{AFX_MSG_MAP(CFinalDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, OnButton1)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFinalDlg message handlers

BOOL CFinalDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon
	
	// TODO: Add extra initialization here
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CFinalDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CFinalDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CFinalDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

void CFinalDlg::OnButton1() 
{
	// TODO: Add your control notification handler code here
	UpdateData(TRUE);

	if(m_count == "3")
	{
				double mtotalArea = 0;

                double LowX = 0.0;
                double LowY = 0.0;
                double MiddleX = 0.0;
                double MiddleY = 0.0;
                double HighX = 0.0;
                double HighY = 0.0;

                double AM = 0.0;
                double BM = 0.0;
                double CM = 0.0;

                double AL = 0.0;
                double BL = 0.0;
                double CL = 0.0;

                double AH = 0.0;
                double BH = 0.0;
                double CH = 0.0;

                double CoefficientL = 0.0;
                double CoefficientH = 0.0;

                double ALtangent = 0.0;
                double BLtangent = 0.0;
                double CLtangent = 0.0;

                double AHtangent = 0.0;
                double BHtangent = 0.0;
                double CHtangent = 0.0;

                double ANormalLine = 0.0;
                double BNormalLine = 0.0;
                double CNormalLine = 0.0;

                double OrientationValue = 0.0;

                double AngleCos = 0.0;

                double Sum1 = 0.0;
                double Sum2 = 0.0;
                double Count2 = 0;
                double Count1 = 0;


                double Sum = 0.0;
                double Radius = 6378000;

                for(int i=0; i<3;i++)
                {
                    if(i==0)
                    {
                        LowX = m_x3 * pi /180;
                        LowY = m_y3 * pi /180;
                        MiddleX = m_x1 * pi /180;
                        MiddleY =m_y1 * pi /180;
                        HighX = m_x2 * pi /180;
                        HighY = m_y2 * pi /180;
                    }

                    else if (i == 2)
                    {
                        LowX = m_x2 * pi /180;
                        LowY = m_y2 * pi /180;
                        MiddleX = m_x3 * pi /180;
                        MiddleY = m_y3 * pi /180;
                        HighX = m_x1 * pi /180;
                        HighY = m_y1 * pi /180;
                    }

                    else
                    {
                        LowX = m_x1 * pi /180;
                        LowY = m_y1 * pi /180;
                        MiddleX = m_x2 * pi /180;
                        MiddleY = m_y2 * pi /180;
                        HighX = m_x3 * pi /180;
                        HighY = m_y3 * pi /180;
                    }

                    AM = cos(MiddleY) * cos(MiddleX);
                    BM = cos(MiddleY) * sin(MiddleX);
                    CM = sin(MiddleY);
                    AL = cos(LowY) * cos(LowX);
                    BL = cos(LowY) * sin(LowX);
                    CL = sin(LowY);
                    AH = cos(HighY) * cos(HighX);
                    BH = cos(HighY) * sin(HighX);
                    CH = sin(HighY);

                    CoefficientL = (AM * AM + BM * BM + CM * CM) / (AM * AL + BM * BL + CM * CL);
                    CoefficientH = (AM * AM + BM * BM + CM * CM) / (AM * AH + BM * BH + CM * CH);

                    ALtangent = CoefficientL * AL - AM;
                    BLtangent = CoefficientL * BL - BM;
                    CLtangent = CoefficientL * CL - CM;
                    AHtangent = CoefficientH * AH - AM;
                    BHtangent = CoefficientH * BH - BM;
                    CHtangent = CoefficientH * CH - CM;

					double temp1 = (AHtangent * ALtangent + BHtangent * BLtangent + CHtangent * CLtangent);
					double temp2 = sqrt(AHtangent * AHtangent + BHtangent * BHtangent + CHtangent * CHtangent) * sqrt(ALtangent * ALtangent + BLtangent * BLtangent + CLtangent * CLtangent);
                    AngleCos = (AHtangent * ALtangent + BHtangent * BLtangent + CHtangent * CLtangent) / (sqrt(AHtangent * AHtangent + BHtangent * BHtangent + CHtangent * CHtangent) * sqrt(ALtangent * ALtangent + BLtangent * BLtangent + CLtangent * CLtangent));

                    AngleCos = acos(AngleCos);

                    ANormalLine = BHtangent * CLtangent - CHtangent * BLtangent;
                    BNormalLine = 0 - (AHtangent * CLtangent - CHtangent * ALtangent);
                    CNormalLine = AHtangent * BLtangent - BHtangent * ALtangent;

                    if (AM != 0)
                        OrientationValue = ANormalLine / AM;
                    else if (BM != 0)
                        OrientationValue = BNormalLine / BM;
                    else
                        OrientationValue = CNormalLine / CM;

                    if (OrientationValue > 0)
                    {
                        Sum1 += AngleCos;
                        Count1++;

                    }
                    else
                    {
                        Sum2 += AngleCos;
                        Count2++;
                        //Sum +=2*Math.PI-AngleCos;
                    }
                }

                if (Sum1 > Sum2)
                {
                    Sum = Sum1 + (2 * pi * Count2 - Sum2);
                }
                else
                {
                    Sum = (2 * pi * Count1 - Sum1) + Sum2;
                }

                //
                mtotalArea = (Sum - (3 - 2) * pi) * Radius * Radius;
				CString s;
				s.Format("%f",mtotalArea);
				AfxMessageBox("答案为："+s);
	}

	else if(m_count == "4")
	{
				double mtotalArea = 0;

                double LowX = 0.0;
                double LowY = 0.0;
                double MiddleX = 0.0;
                double MiddleY = 0.0;
                double HighX = 0.0;
                double HighY = 0.0;

                double AM = 0.0;
                double BM = 0.0;
                double CM = 0.0;

                double AL = 0.0;
                double BL = 0.0;
                double CL = 0.0;

                double AH = 0.0;
                double BH = 0.0;
                double CH = 0.0;

                double CoefficientL = 0.0;
                double CoefficientH = 0.0;

                double ALtangent = 0.0;
                double BLtangent = 0.0;
                double CLtangent = 0.0;

                double AHtangent = 0.0;
                double BHtangent = 0.0;
                double CHtangent = 0.0;

                double ANormalLine = 0.0;
                double BNormalLine = 0.0;
                double CNormalLine = 0.0;

                double OrientationValue = 0.0;

                double AngleCos = 0.0;

                double Sum1 = 0.0;
                double Sum2 = 0.0;
                double Count2 = 0;
                double Count1 = 0;


                double Sum = 0.0;
                double Radius = 6378000;

                for(int i=0; i<4;i++)
                {
                    if(i==0)
                    {
                        LowX = m_x4 * pi /180;
                        LowY = m_y4 * pi /180;
                        MiddleX = m_x1 * pi /180;
                        MiddleY =m_y1 * pi /180;
                        HighX = m_x2 * pi /180;
                        HighY = m_y2 * pi /180;
                    }

                    else if (i == 3)
                    {
                        LowX = m_x3 * pi /180;
                        LowY = m_y3 * pi /180;
                        MiddleX = m_x4 * pi /180;
                        MiddleY = m_y4 * pi /180;
                        HighX = m_x1 * pi /180;
                        HighY = m_y1 * pi /180;
                    }

					else if (i == 2)
                    {
                        LowX = m_x2 * pi /180;
                        LowY = m_y2 * pi /180;
                        MiddleX = m_x3 * pi /180;
                        MiddleY = m_y3 * pi /180;
                        HighX = m_x4 * pi /180;
                        HighY = m_y4 * pi /180;
                    }

                    else
                    {
                        LowX = m_x1 * pi /180;
                        LowY = m_y1 * pi /180;
                        MiddleX = m_x2 * pi /180;
                        MiddleY = m_y2 * pi /180;
                        HighX = m_x3 * pi /180;
                        HighY = m_y3 * pi /180;
                    }

                    AM = cos(MiddleY) * cos(MiddleX);
                    BM = cos(MiddleY) * sin(MiddleX);
                    CM = sin(MiddleY);
                    AL = cos(LowY) * cos(LowX);
                    BL = cos(LowY) * sin(LowX);
                    CL = sin(LowY);
                    AH = cos(HighY) * cos(HighX);
                    BH = cos(HighY) * sin(HighX);
                    CH = sin(HighY);

                    CoefficientL = (AM * AM + BM * BM + CM * CM) / (AM * AL + BM * BL + CM * CL);
                    CoefficientH = (AM * AM + BM * BM + CM * CM) / (AM * AH + BM * BH + CM * CH);

                    ALtangent = CoefficientL * AL - AM;
                    BLtangent = CoefficientL * BL - BM;
                    CLtangent = CoefficientL * CL - CM;
                    AHtangent = CoefficientH * AH - AM;
                    BHtangent = CoefficientH * BH - BM;
                    CHtangent = CoefficientH * CH - CM;

					double temp1 = (AHtangent * ALtangent + BHtangent * BLtangent + CHtangent * CLtangent);
					double temp2 = sqrt(AHtangent * AHtangent + BHtangent * BHtangent + CHtangent * CHtangent) * sqrt(ALtangent * ALtangent + BLtangent * BLtangent + CLtangent * CLtangent);
                    AngleCos = (AHtangent * ALtangent + BHtangent * BLtangent + CHtangent * CLtangent) / (sqrt(AHtangent * AHtangent + BHtangent * BHtangent + CHtangent * CHtangent) * sqrt(ALtangent * ALtangent + BLtangent * BLtangent + CLtangent * CLtangent));

                    AngleCos = acos(AngleCos);

                    ANormalLine = BHtangent * CLtangent - CHtangent * BLtangent;
                    BNormalLine = 0 - (AHtangent * CLtangent - CHtangent * ALtangent);
                    CNormalLine = AHtangent * BLtangent - BHtangent * ALtangent;

                    if (AM != 0)
                        OrientationValue = ANormalLine / AM;
                    else if (BM != 0)
                        OrientationValue = BNormalLine / BM;
                    else
                        OrientationValue = CNormalLine / CM;

                    if (OrientationValue > 0)
                    {
                        Sum1 += AngleCos;
                        Count1++;

                    }
                    else
                    {
                        Sum2 += AngleCos;
                        Count2++;
                        //Sum +=2*Math.PI-AngleCos;
                    }
                }

                if (Sum1 > Sum2)
                {
                    Sum = Sum1 + (2 * pi * Count2 - Sum2);
                }
                else
                {
                    Sum = (2 * pi * Count1 - Sum1) + Sum2;
                }

                //
                mtotalArea = (Sum - (4 - 2) * pi) * Radius * Radius;
				CString s;
				s.Format("%f",mtotalArea);
				AfxMessageBox("答案为："+s);
	}

	else
	{
		AfxMessageBox("对不起，您输入的坐标点数不够建立模型");
		GetDlgItem(IDC_BUTTON1)->SetFocus();
	}
	
	
}
