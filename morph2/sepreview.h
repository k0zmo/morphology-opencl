#pragma once

#include <QDialog>
#include <opencv2/core/core.hpp>

class PreviewLabel : public QLabel
{
	Q_OBJECT
public:
	PreviewLabel(QWidget* parent = nullptr);
	void setPreviewImage(const cv::Mat& se);
	virtual void mousePressEvent(QMouseEvent* evt);

private:
	QSizeF pixSize;
	cv::Mat se;

signals:
	void structuringElementModified(const cv::Mat& se);
};

#include "ui_sepreview.h"

class StructuringElementPreview : public QDialog, Ui::SEPreview
{
	Q_OBJECT
public:
	StructuringElementPreview(QWidget* parent = 0);
	virtual void closeEvent(QCloseEvent* evt);

public slots:
	void onStructuringElementChanged(const cv::Mat& se);

signals:
	void closed();
};