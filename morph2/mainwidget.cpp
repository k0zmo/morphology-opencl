#include "mainwidget.h"
#include "controller.h"

MainWidget::MainWidget(QWidget* parent)
	: QWidget(parent)
	, disableRecomputing(false)
	, spacer(new QSpacerItem(0, 0, 
			QSizePolicy::Minimum, 
			QSizePolicy::MinimumExpanding))
	, cameraOn(false)
{
	setupUi(this);

	// Operacje
	connect(rbNone, SIGNAL(toggled(bool)), SLOT(onOperationToggled(bool)));
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
	connect(rbRect, SIGNAL(toggled(bool)), SLOT(onElementTypeToggled(bool)));
	connect(rbEllipse, SIGNAL(toggled(bool)), SLOT(onElementTypeToggled(bool)));
	connect(rbCross, SIGNAL(toggled(bool)), SLOT(onElementTypeToggled(bool)));
	connect(rbCustom, SIGNAL(toggled(bool)), SLOT(onElementTypeToggled(bool)));

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

	// Wartosci domyslne
	rbNone->toggle();
	rbEllipse->toggle();
	cbSquare->setChecked(true);
	lbXElementSize->setText(QString::fromLatin1("Horizontal: 3"));
	lbYElementSize->setText(QString::fromLatin1("Vertical: 3"));
	lbRotation->setText(QString::fromLatin1("0"));
}

MainWidget::~MainWidget()
{
}

cvu::EMorphOperation MainWidget::morphologyOperation() const
{
	using namespace cvu;

	if(rbErode->isChecked())              { return MO_Erode; }
	else if(rbDilate->isChecked())        { return MO_Dilate; }
	else if(rbOpen->isChecked())          { return MO_Open; }
	else if(rbClose->isChecked())         { return MO_Close; }
	else if(rbGradient->isChecked())      { return MO_Gradient; }
	else if(rbTopHat->isChecked())        { return MO_TopHat; }
	else if(rbBlackHat->isChecked())      { return MO_BlackHat; }
	else if(rbOutline->isChecked())       { return MO_Outline; }
	else if(rbSkeleton->isChecked())      { return MO_Skeleton; }
	else if(rbSkeletonZhang->isChecked()) { return MO_Skeleton_ZhangSuen; }
	else                                  { return MO_None; }
}

cvu::EStructuringElementType MainWidget::structuringElementType() const
{
	using namespace cvu;

	if(rbRect->isChecked())         return SET_Rect;
	else if(rbEllipse->isChecked()) return SET_Ellipse;
	else if(rbCross->isChecked())   return SET_Cross;
	else                            return SET_Custom;
}

void MainWidget::setStructuringElementType(
	cvu::EStructuringElementType type)
{
	switch(type)
	{
	case cvu::SET_Rect:
		rbRect->setChecked(true); break;
	case cvu::SET_Ellipse:
		rbEllipse->setChecked(true); break;
	case cvu::SET_Cross:
		rbCross->setChecked(true); break;
	case cvu::SET_Custom:
		rbCustom->setChecked(true); break;
	}
}

QSize MainWidget::structuringElementSize() const
{
	return QSize(
		hsXElementSize->value(),
		hsYElementSize->value()
	);
}

void MainWidget::setStructuringElementSize(const QSize& size)
{
	if(size.width() != size.height())
		cbSquare->setChecked(false);
	else
		cbSquare->setChecked(true);

	hsXElementSize->setValue(size.width());
	hsYElementSize->setValue(size.height());
}

int MainWidget::structuringElementRotation() const
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

void MainWidget::setStructuringElementRotation(int angle)
{
	if(angle >= 180)
		angle += 180;
	else
		angle -= 180;

	angle = 360 - angle;
	angle = angle % 360;

	dialRotation->setValue(angle);
}

QRadioButton* MainWidget::operationToRadioBox(cvu::EMorphOperation op)
{
	using namespace cvu;

	switch(op)
	{
	case MO_None:     return rbNone;
	case MO_Erode:    return rbErode;
	case MO_Dilate:   return rbDilate;
	case MO_Open:     return rbOpen;
	case MO_Close:    return rbClose;
	case MO_Gradient: return rbGradient;
	case MO_TopHat:   return rbTopHat;
	case MO_BlackHat: return rbBlackHat;
	case MO_Outline:  return rbOutline;
	case MO_Skeleton: return rbSkeleton;
	case MO_Skeleton_ZhangSuen: return rbSkeletonZhang;
	default:          return rbNone;
	}
}

void MainWidget::onOperationToggled(bool checked)
{
	if(!checked)
		return;

	// Dla braku operacji
	//pbRun->setEnabled(!rbNone->isChecked());

	// Deaktywuj wybor elementu strukturalnego
	// dla operacji hitmiss oraz braku operacji
	if (rbOutline->isChecked() ||
		rbSkeleton->isChecked() ||
		rbSkeletonZhang->isChecked() ||
		rbNone->isChecked())
	{
		gbElement->setEnabled(false);
		gbElementSize->setEnabled(false);
	}
	else
	{
		// Aktywuj wybor elementu strukturalnego
		gbElement->setEnabled(true);
		gbElementSize->setEnabled(true);
	}

	if(cbAutoTrigger->isChecked() || 
	   rbNone->isChecked() ||
	   cameraOn)
	{
		emit recomputeNeeded();
	}
}

void MainWidget::onElementTypeToggled(bool checked)
{
	if(!checked)
		return;

	emit structuringElementChanged();

	if((cbAutoTrigger->isChecked() && !rbNone->isChecked()) ||
	   cameraOn)
	{
		emit recomputeNeeded();
	}
}

void MainWidget::onElementSizeRatioChanged(int state)
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

void MainWidget::onElementSizeChanged(int value)
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
	if (!disableRecomputing)
	{
		if((cbAutoTrigger->isChecked() && !rbNone->isChecked()) ||
		   cameraOn)
		{
			emit recomputeNeeded();
		}
	}		

	disableRecomputing = false;
}

void MainWidget::onElementRotationChanged(int value)
{
	Q_UNUSED(value);

	int angle = structuringElementRotation();
	lbRotation->setText(QString::number(angle));

	emit structuringElementChanged();

	if((cbAutoTrigger->isChecked() && !rbNone->isChecked()) ||
	   cameraOn)
	{
		emit recomputeNeeded();
	}
}

void MainWidget::onElementRotationResetPressed()
{
	setStructuringElementRotation(0);
}
