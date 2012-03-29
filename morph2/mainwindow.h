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

	// Zwraca wybrany indeks filtru bajera (0 dla zadnego)
	int bayerIndex() const
	{ return cmbBayer->currentIndex(); }

	// Zwraca wybrana operacje morfologiczna
	Morphology::EOperationType morphologyOperation() const;
	// Ustawia wybrana operacje morfologiczna
	void setMorphologyOperation(Morphology::EOperationType op)
	{ operationToRadioBox(op)->setChecked(true); }

	// Czy wybrano brak operacji (morfologicznej)
	bool isNoneOperationChecked() const
	{ return rbNone->isChecked(); }

	// Zwraca wybrany typ elementu strukturalnego
	Morphology::EStructuringElementType structuringElementType() const;
	// Ustawia typ elementu strukturalnego
	void setStructuringElementType(Morphology::EStructuringElementType type);

	// Zwraca wybrany rozmiar elementu strukturalnego
	QSize structuringElementSize() const;
	// Ustawia rozmiar elementu strukturalnego
	void setStructuringElementSize(const QSize& size);

	// Zwraca wybrana rotacje elementu strukturalnego
	int structuringElementRotation() const;
	// Ustawia rotacje elementu strukturalnego
	void setStructuringElementRotation(int rotation);

	// Ustawia tekst na pasku stanu 
	void setStatusBarText(const QString& message)
	{ statusBarLabel->setText(message); }

	void setCameraStatusBarState(bool connected)
	{ 
		cameraStatusLabel->setText(connected ? 
			"Camera: Connected" : "Camera: Not connected");
	}

	// Ustawia mozliwosc zaznaczenia "silnika" OpenCL
	void setOpenCLCheckableAndChecked(bool state);

	// Ustawia tekst na przycisku `Show structuring element`
	void setStructuringElementPreviewButtonText(const QString& text)
	{ pbShowSE->setText(text); }

	// Ustawia dostepnosc kontrolki odpowiadajacej za wybor rotacji el. strukturalnego
	void setEnabledStructuringElementRotation(bool state)
	{ 
		dialRotation->setEnabled(state);
		pbResetRotation->setEnabled(state);
	}

	void setFromCamera(bool state)
	{ actionCameraInput->setChecked(state); }

	void setEnabledSaveOpenFile(bool state)
	{
		actionOpen->setEnabled(state);
		actionSave->setEnabled(state);
	}

private:
	//Zwraca kontrolke reprezentujaca daneaoperacje morfologiczna
	QRadioButton* operationToRadioBox(Morphology::EOperationType op);

private:
	bool disableRecomputing;
	QLabel* statusBarLabel;
	QLabel* cameraStatusLabel;
	QSpacerItem* spacer;

private slots:
	void onOperationToggled(bool checked);
	void onElementTypeToggled(bool checked);
	void onElementSizeRatioChanged(int state);
	void onElementSizeChanged(int value);
	void onElementRotationChanged(int value);
	void onElementRotationResetPressed();

signals:
	void structuringElementChanged();
	void recomputeNeeded();
};