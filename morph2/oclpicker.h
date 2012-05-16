#pragma once

#include <QDialog>

#include "ui_oclpicker.h"
#include "oclcontext.h"
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
	void setInteropEnabled(bool state)
	{ tryInteropCheckBox->setEnabled(state); }

private slots:
	void onItemSelectionChanged();
	void accept();
	void onTryInteropToggled(bool checked);

private:
	QMap<QString, QString> devToDesc;
	int platformId;
	int deviceId;
	bool interop;
};

