#pragma once

#include <QWidget>
#include "ui_mainwidget.h"

#include "morphop.h"

class MainWidget : public QWidget, Ui::MainWidget
{
	Q_OBJECT
public:
	MainWidget(QWidget* parent = 0);
	virtual ~MainWidget();

	// Zwraca wybrany indeks filtru bajera (0 dla zadnego)
	int bayerIndex() const
	{ return cmbBayer->currentIndex(); }

	// Zwraca wybrana operacje morfologiczna
	cvu::EMorphOperation morphologyOperation() const;
	// Ustawia wybrana operacje morfologiczna
	void setMorphologyOperation(cvu::EMorphOperation op)
	{ operationToRadioBox(op)->setChecked(true); }

	// Czy wybrano brak operacji (morfologicznej)
	bool isNoneOperationChecked() const
	{ return rbNone->isChecked(); }

	// Zwraca wybrany typ elementu strukturalnego
	cvu::EStructuringElementType structuringElementType() const;
	// Ustawia typ elementu strukturalnego
	void setStructuringElementType(cvu::EStructuringElementType type);

	// Zwraca wybrany rozmiar elementu strukturalnego
	QSize structuringElementSize() const;
	// Ustawia rozmiar elementu strukturalnego
	void setStructuringElementSize(const QSize& size);

	// Zwraca wybrana rotacje elementu strukturalnego
	int structuringElementRotation() const;
	// Ustawia rotacje elementu strukturalnego
	void setStructuringElementRotation(int rotation);

	// Ustawia tekst na przycisku `Show structuring element`
	void setStructuringElementPreviewButtonText(const QString& text)
	{ pbShowSE->setText(text); }

	// Ustawia dostepnosc kontrolki odpowiadajacej za wybor rotacji el. strukturalnego
	void setEnabledStructuringElementRotation(bool state)
	{ 
		dialRotation->setEnabled(state);
		pbResetRotation->setEnabled(state);
	}

	void setEnabledAutoRecompute(bool state)
	{ cbAutoTrigger->setEnabled(state); }

	void setCameraStatus(bool state)
	{ cameraOn = state; }

private:
	//Zwraca kontrolke reprezentujaca daneaoperacje morfologiczna
	QRadioButton* operationToRadioBox(cvu::EMorphOperation op);

private:
	bool disableRecomputing;
	QSpacerItem* spacer;
	bool cameraOn;

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
