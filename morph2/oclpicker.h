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

private slots:
	void onItemSelectionChanged();
	void accept();

private:
	QMap<QString, QString> devToDesc;
	int platformId;
	int deviceId;
};

