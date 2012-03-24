#pragma once

#include <QDialog>

#include "ui_settings.h"
#include "configuration.h"

class Settings : public QDialog, Ui::SettingDialog
{
	Q_OBJECT
public:
	Settings(QWidget* parent = 0);

	void setConfigurationModel(const Configuration& conf);
	Configuration configurationModel() const;
};