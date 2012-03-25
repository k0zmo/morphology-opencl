#include "mainwindow.h"
#include "controller.h"


MainWindow::MainWindow(QWidget* parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
	, disableRecomputing(false)
	, spacer(new QSpacerItem(0, 0, 
			QSizePolicy::Minimum, 
			QSizePolicy::MinimumExpanding))
{
	setupUi(this);

	// Menu (File)
	connect(actionCameraInput, SIGNAL(triggered(bool)), gC, SLOT(onFromCameraTriggered(bool)));
	connect(actionOpen, SIGNAL(triggered()), gC, SLOT(onOpenFileTriggered()));
	connect(actionSave, SIGNAL(triggered()), gC, SLOT(onSaveFileTriggered()));
	connect(actionOpenSE, SIGNAL(triggered()), gC, SLOT(onOpenStructuringElementTriggered()));
	connect(actionSaveSE, SIGNAL(triggered()), gC, SLOT(onSaveStructuringElementTriggered()));
	connect(actionExit, SIGNAL(triggered()), SLOT(close()));

	// Menu Settings
	connect(actionOpenCL, SIGNAL(triggered(bool)), gC, SLOT(onOpenCLTriggered(bool)));
	connect(actionPickMethod, SIGNAL(triggered()), gC, SLOT(onPickMethodTriggerd()));
	connect(actionSettings, SIGNAL(triggered()), gC, SLOT(onSettingsTriggered()));

	// Operacje
	connect(rbNone, SIGNAL(toggled(bool)), SLOT(onNoneOperationToggled(bool)));
	connect(rbErode, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbDilate, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbOpen, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbClose, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbGradient, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbTopHat, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbBlackHat, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbOutline, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbSkeleton, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
	connect(rbSkeletonZhang, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));

	// Element strukturalny
	connect(rbRect, SIGNAL(toggled(bool)), SLOT(onStructuringElementToggled(bool)));
	connect(rbEllipse, SIGNAL(toggled(bool)), SLOT(onStructuringElementToggled(bool)));
	connect(rbCross, SIGNAL(toggled(bool)), SLOT(onStructuringElementToggled(bool)));
	connect(rbCustom, SIGNAL(toggled(bool)), SLOT(onStructuringElementToggled(bool)));

	// Rozmiar elementu strukturalnego
	connect(cbSquare, SIGNAL(stateChanged(int)), SLOT(onElementSizeRatioChanged(int)));
	connect(hsXElementSize, SIGNAL(valueChanged(int)), SLOT(onElementSizeChanged(int)));
	connect(hsYElementSize, SIGNAL(valueChanged(int)), SLOT(onElementSizeChanged(int)));
	connect(dialRotation, SIGNAL(valueChanged(int)), SLOT(onElementRotationChanged(int)));
	connect(pbResetRotation, SIGNAL(pressed()), SLOT(onElementRotationResetPressed()));

	connect(cbAutoTrigger, SIGNAL(stateChanged(int)), gC, SLOT(onAutoTriggerChanged(int)));
	connect(cbInvert, SIGNAL(stateChanged(int)), gC, SLOT(onInvertChanged(int)));
	connect(cmbBayer, SIGNAL(currentIndexChanged(int)), gC, SLOT(onBayerIndexChanged(int)));
	connect(pbShowSE, SIGNAL(pressed()), gC, SLOT(onStructuringElementPreviewPressed()));
	connect(pbRun, SIGNAL(pressed()), gC, SLOT(onRecompute()));

	cbInvert->setVisible(false);

	// Wartosci domyslne
	rbNone->toggle();
	rbEllipse->toggle();
	cbSquare->setChecked(true);

	lbXElementSize->setText(QString::fromLatin1("Horizontal: 3"));
	lbYElementSize->setText(QString::fromLatin1("Vertical: 3"));
	lbRotation->setText(QString::fromLatin1("0"));

	statusBarLabel = new QLabel(this);
	statusBar()->addPermanentWidget(statusBarLabel);
}

MainWindow::~MainWindow()
{
}

void MainWindow::setPreviewWidget(QWidget* previewWidget)
{
	if(!previewVerticalLayout->isEmpty())
	{
		previewVerticalLayout->removeWidget(previewWidget);
		previewVerticalLayout->removeItem(spacer);
	}

	previewVerticalLayout->addWidget(previewWidget);
	previewVerticalLayout->addItem(spacer);
}

Morphology::EOperationType MainWindow::morphologyOperation() const
{
	using namespace Morphology;

	if(rbErode->isChecked())              { return OT_Erode; }
	else if(rbDilate->isChecked())        { return OT_Dilate; }
	else if(rbOpen->isChecked())          { return OT_Open; }
	else if(rbClose->isChecked())         { return OT_Close; }
	else if(rbGradient->isChecked())      { return OT_Gradient; }
	else if(rbTopHat->isChecked())        { return OT_TopHat; }
	else if(rbBlackHat->isChecked())      { return OT_BlackHat; }
	else if(rbOutline->isChecked())       { return OT_Outline; }
	else if(rbSkeleton->isChecked())      { return OT_Skeleton; }
	else if(rbSkeletonZhang->isChecked()) { return OT_Skeleton_ZhangSuen; }
	else                                  { return OT_None; }
}

Morphology::EStructuringElementType MainWindow::structuringElementType() const
{
	using namespace Morphology;

	if(rbRect->isChecked())         return SET_Rect;
	else if(rbEllipse->isChecked()) return SET_Ellipse;
	else if(rbCross->isChecked())   return SET_Cross;
	else                            return SET_Custom;
}

void MainWindow::setStructuringElementType(
	Morphology::EStructuringElementType type)
{
	switch(type)
	{
	case Morphology::SET_Rect:
		rbRect->setChecked(true); break;
	case Morphology::SET_Ellipse:
		rbEllipse->setChecked(true); break;
	case Morphology::SET_Cross:
		rbCross->setChecked(true); break;
	case Morphology::SET_Custom:
		rbCustom->setChecked(true); break;
	}
}

QSize MainWindow::structuringElementSize() const
{
	return QSize(
		hsXElementSize->value(),
		hsYElementSize->value()
	);
}

void MainWindow::setStructuringElementSize(const QSize& size)
{
	if(size.width() != size.height())
		cbSquare->setChecked(false);
	else
		cbSquare->setChecked(true);

	hsXElementSize->setValue(size.width());
	hsYElementSize->setValue(size.height());
}

int MainWindow::structuringElementRotation() const
{
	int angle = dialRotation->value();

	if(angle >= 180) 
		angle -= 180;
	else 
		angle += 180;

	angle = 360 - angle;
	angle = angle % 360;

	return angle;
}

void MainWindow::setStructuringElementRotation(int angle)
{
	if(angle >= 180)
		angle += 180;
	else
		angle -= 180;

	angle = 360 - angle;
	angle = angle % 360;

	dialRotation->setValue(angle);
}

QRadioButton* MainWindow::operationToRadioBox(Morphology::EOperationType op)
{
	using namespace Morphology;

	switch(op)
	{
	case OT_None:     return rbNone;
	case OT_Erode:    return rbErode;
	case OT_Dilate:   return rbDilate;
	case OT_Open:     return rbOpen;
	case OT_Close:    return rbClose;
	case OT_Gradient: return rbGradient;
	case OT_TopHat:   return rbTopHat;
	case OT_BlackHat: return rbBlackHat;
	case OT_Outline:  return rbOutline;
	case OT_Skeleton: return rbSkeleton;
	case OT_Skeleton_ZhangSuen: rbSkeletonZhang;
	default:          return rbNone;
	}
}

void MainWindow::onNoneOperationToggled(bool checked)
{
	if(!checked)
	{
		pbRun->setEnabled(true);
		return;
	}

	gbElement->setEnabled(true);
	gbElementSize->setEnabled(true);
	pbRun->setEnabled(false);
	actionSave->setEnabled(false);

	emit sourceImageShowed();
}

void MainWindow::onOperationToggled(bool checked)
{
	if(!checked)
		return;

	// Operacje hit-miss
	if (rbOutline->isChecked() ||
		rbSkeleton->isChecked() ||
		rbSkeletonZhang->isChecked())
	{
		// deaktywuj wybor elementu strukturalnego
		gbElement->setEnabled(false);
		gbElementSize->setEnabled(false);
	}
	else
	{
		// aktywuj wybor elementu strukturalnego
		gbElement->setEnabled(true);
		gbElementSize->setEnabled(true);
	}

	if(cbAutoTrigger->isChecked())
		emit recomputeNeeded();
}

void MainWindow::onStructuringElementToggled(bool checked)
{
	if(!checked)
		return;

	emit structuringElementChanged();

	if(cbAutoTrigger->isChecked())
		emit recomputeNeeded();
}

void MainWindow::onElementSizeRatioChanged(int state)
{
	if(state != Qt::Checked)
		return;

	int vv = qMax(
		hsXElementSize->value(), 
		hsYElementSize->value());

	hsXElementSize->setValue(vv);
	hsYElementSize->setValue(vv);

	emit structuringElementChanged();
}

void MainWindow::onElementSizeChanged(int value)
{
	QSlider* notifier = qobject_cast<QSlider*>(sender());
	QSlider* fbslider;

	// Zdecyduj dla ktorej kontrolki wywolano zdarzenie
	if(notifier == hsXElementSize)
	{
		lbXElementSize->setText(QLatin1String("Horizontal: ") + 
			QString::number(2 * value + 1));
		fbslider = hsYElementSize;
	}
	else
	{
		lbYElementSize->setText(QString::fromLatin1("Vertical: ") + 
			QString::number(2 * value + 1));
		fbslider = hsXElementSize;
	}

	// Jesli mamy zaznaczone 1:1 ratio
	if (cbSquare->checkState() == Qt::Checked)
	{
		if (fbslider->value() != value)
		{
			disableRecomputing = true;
			fbslider->setValue(value);
		}
	}

	emit structuringElementChanged();

	// Czy trzeba ponownie wykonac obliczenia w zwiazku ze zmiana 
	// rozmiaru elementu strukturalnego
	if (!disableRecomputing && cbAutoTrigger->isChecked())
		emit recomputeNeeded();

	disableRecomputing = false;
}

void MainWindow::onElementRotationChanged(int value)
{
	Q_UNUSED(value);

	int angle = structuringElementRotation();
	lbRotation->setText(QString::number(angle));

	emit structuringElementChanged();

	if(cbAutoTrigger->isChecked())
		emit recomputeNeeded();
}

void MainWindow::onElementRotationResetPressed()
{
	setStructuringElementRotation(0);
}

void MainWindow::setOpenCLCheckableAndChecked(bool state)
{
	actionOpenCL->setEnabled(state);
	actionOpenCL->setChecked(state);
}
