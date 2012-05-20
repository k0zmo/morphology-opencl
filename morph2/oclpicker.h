#pragma once

#include <QDialog>

#include "ui_oclpicker.h"
#include "oclthread.h"

class oclPicker : public QDialog, Ui::oclPicker
{
	Q_OBJECT
public:
	explicit oclPicker(const PlatformDevicesMap& map,
		QWidget* parent = 0);
	~oclPicker();

	int platform() const { return platformId; }
	int device() const { return deviceId; }
	bool tryInterop() const { return interop; }
	EOpenCLBackend openclBackend() const { return backend; }

private slots:
	void accept();

	void onItemSelectionChanged();
	//void onTryInteropToggled(bool checked);
	void onFilteringChanged();

private:
	QMap<QString, QString> devToDesc;
	int platformId;
	int deviceId;
	bool interop;
	EOpenCLBackend backend;
};

