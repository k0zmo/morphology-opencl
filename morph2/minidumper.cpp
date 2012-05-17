#include "minidumper.h"

#ifdef Q_WS_WIN32

typedef BOOL (WINAPI *MINIDUMPWRITEDUMP)(HANDLE hProcess, DWORD dwPid, HANDLE hFile, MINIDUMP_TYPE DumpType,
	CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
	CONST PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
	CONST PMINIDUMP_CALLBACK_INFORMATION CallbackParam);
	
LPCSTR MiniDumper::m_szAppName;

MiniDumper::MiniDumper(LPCSTR szAppName)
{
	assert(m_szAppName == NULL);

	m_szAppName = szAppName ? strdup(szAppName) : "Application";

	::SetUnhandledExceptionFilter(TopLevelFilter);
}

LONG MiniDumper::TopLevelFilter(struct _EXCEPTION_POINTERS *pExceptionInfo)
{
	LONG retval = EXCEPTION_CONTINUE_SEARCH;

	// firstly see if dbghelp.dll is around and has the function we need
	// look next to the EXE first, as the one in System32 might be old 
	// (e.g. Windows 2000)
	HMODULE hDll = NULL;
	char szDbgHelpPath[_MAX_PATH];

	if(GetModuleFileNameA(NULL, szDbgHelpPath, _MAX_PATH))
	{
		char *pSlash = _tcsrchr(szDbgHelpPath, '\\');
		if(pSlash)
		{
			_tcscpy(pSlash+1, "dbghelp.dll");
			hDll = ::LoadLibraryA(szDbgHelpPath);
		}
	}

	if(hDll==NULL)
	{
		// load any version we can
		hDll = ::LoadLibraryA("dbghelp.dll");
	}

	LPCSTR szResult = NULL;

	if(hDll)
	{
		MINIDUMPWRITEDUMP pDump = reinterpret_cast<MINIDUMPWRITEDUMP>
			(::GetProcAddress(hDll, "MiniDumpWriteDump"));

		if(pDump)
		{
			char szDumpPath[_MAX_PATH];
			char szScratch [_MAX_PATH];

			_tcscpy(szDumpPath, m_szAppName);
			_tcscat(szDumpPath, ".dmp");

			// ask the user if they want to save a dump file
			if(::MessageBoxA(NULL, 
				"Something bad happened in your program, would you like to save a diagnostic file?", 
				m_szAppName, MB_YESNO) == IDYES)
			{
				// create the file
				HANDLE hFile = ::CreateFileA(szDumpPath, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS,
					FILE_ATTRIBUTE_NORMAL, NULL);

				if(hFile != INVALID_HANDLE_VALUE)
				{
					_MINIDUMP_EXCEPTION_INFORMATION ExInfo;

					ExInfo.ThreadId = ::GetCurrentThreadId();
					ExInfo.ExceptionPointers = pExceptionInfo;
					ExInfo.ClientPointers = NULL;

					MINIDUMP_TYPE mdt = MINIDUMP_TYPE(
						MiniDumpWithPrivateReadWriteMemory |
						MiniDumpWithDataSegs |
						MiniDumpWithHandleData |
						MiniDumpWithFullMemoryInfo |
						MiniDumpWithThreadInfo |
						MiniDumpWithUnloadedModules);
					//MINIDUMP_TYPE mdt = MiniDumpNormal;

					// write the dump
					BOOL bOK = pDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, mdt, &ExInfo, NULL, NULL);
					if(bOK)
					{
						sprintf(szScratch, "Saved dump file to '%s'", szDumpPath);
						szResult = szScratch;
						retval = EXCEPTION_EXECUTE_HANDLER;
					}
					else
					{
						sprintf(szScratch, "Failed to save dump file to '%s' (error 0x%x)", szDumpPath, GetLastError());
						szResult = szScratch;
					}
					::CloseHandle(hFile);
				}
				else
				{
					sprintf(szScratch, "Failed to create dump file '%s' (error 0x%x)", szDumpPath, GetLastError());
					szResult = szScratch;
				}
			}
		}
		else
		{
			szResult = "DBGHELP.DLL too old";
		}
	}
	else
	{
		szResult = "DBGHELP.DLL not found";
	}

	if (szResult)
		::MessageBoxA(NULL, szResult, m_szAppName, MB_OK);

	return retval;
}

#endif