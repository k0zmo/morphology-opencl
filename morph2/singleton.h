#pragma once

#include <qglobal.h>

template <typename T>
class Singleton
{
private:
	Singleton(const Singleton<T>&);
	Singleton& operator=(const Singleton<T>&);

protected:
	static T* msSingleton;

public:
	Singleton()
	{ Q_ASSERT(!msSingleton); msSingleton = static_cast<T*>(this); }

	~Singleton()
	{ Q_ASSERT(msSingleton); msSingleton = 0; }

	static T& getSingleton()
	{ Q_ASSERT(msSingleton); return *msSingleton; }

	static T* getSingletonPtr()
	{ return msSingleton; }
};