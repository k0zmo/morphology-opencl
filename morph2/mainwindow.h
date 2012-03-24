#pragma once

#include <QtGui/QMainWindow>
#include "ui_mainwindow.h"

#include "morphoperators.h"

class MainWindow : public QMainWindow, Ui::MainWindow
{
	Q_OBJECT
public:
	MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
	virtual ~MainWindow();

	void setPreviewWidget(QWidget* previewWidget);

	void allowImageSave()
	{ actionSave->setEnabled(true); }

	int bayerIndex() const
	{ return cmbBayer->currentIndex(); }

	Morphology::EOperationType morphologyOperation() const;
	void setMorphologyOperation(Morphology::EOperationType op)
	{ operationToRadioBox(op)->setChecked(true); }

	bool isNoneOperationChecked() const
	{ return rbNone->isChecked(); }

	Morphology::EStructuringElementType structuringElementType() const;
	void setStructuringElementType(Morphology::EStructuringElementType type);

	QSize structuringElementSize() const;
	void setStructuringElementSize(const QSize& size);

	int structuringElementRotation() const;
	void setStructuringElementRotation(int rotation);

	void setStatusBarText(const QString& message)
	{ statusBarLabel->setText(message); }

	void setOpenCLCheckableAndChecked(bool state);

private:
	QRadioButton* operationToRadioBox(Morphology::EOperationType op);

private:
	bool disableRecomputing;
	QLabel* statusBarLabel;
	QSpacerItem* spacer;

private slots:
	void onNoneOperationToggled(bool checked);
	void onOperationToggled(bool checked);
	void onStructuringElementToggled(bool checked);
	void onElementSizeRatioChanged(int state);
	void onElementSizeChanged(int value);
	void onElementRotationChanged(int value);
	void onElementRotationResetPressed();

signals:
	void structuringElementChanged();
	void sourceImageShowed();
	void recomputeNeeded();
};