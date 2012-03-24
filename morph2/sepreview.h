#pragma once

#include <QDialog>
#include <opencv2/core/core.hpp>

#include "ui_sepreview.h"

class SEPreview : public QDialog, Ui::SEPreview
{
	Q_OBJECT
public:
	SEPreview(QWidget* parent = 0);

public slots:
	void onStructuringElementChanged(const cv::Mat& se);
};