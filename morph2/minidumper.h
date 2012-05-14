#pragma once

#ifdef Q_WS_WIN32

#include <dbghelp.h>
#include <tchar.h>

class MiniDumper
{
private:
	static LPCSTR m_szAppName;
	static LONG WINAPI TopLevelFilter
		(struct _EXCEPTION_POINTERS *pExceptionInfo);

public:
	MiniDumper(LPCSTR szAppName);
};

#endif 