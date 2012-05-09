#include "settings.h"

Settings::Settings(QWidget* parent)
	: QDialog(parent)
{
	setupUi(this);

	QIntValidator* intv = new QIntValidator(0, 8192, this);
	maxImageWidthLineEdit->setValidator(intv);
	maxImageHeightLineEdit->setValidator(intv);
}

void Settings::setConfigurationModel(const Configuration& conf)
{
	auto setComboBoxIndex = [](QComboBox* cb, const QString& v)
	{
		int i = cb->findText(v);
		cb->setCurrentIndex(i);
	};

	// Sekcja Preview
	maxImageWidthLineEdit->setText(QString::number(conf.maxImageWidth));
	maxImageHeightLineEdit->setText(QString::number(conf.maxImageHeight));
	defaultImageLineEdit->setText(conf.defaultImage);

	// Sekcja OpenCL
	useAtomicCountersCheckBox->setChecked(conf.atomicCounters);
	setComboBoxIndex(workgroupSizeXComboBox, QString::number(conf.workgroupSizeX));
	setComboBoxIndex(workgroupSizeYComboBox, QString::number(conf.workgroupSizeY));

	// Sekcja Buffer 2D kernels
	setComboBoxIndex(erodeKernelComboBox, conf.erode_2d);
	setComboBoxIndex(dilateKernelComboBox, conf.dilate_2d);
	setComboBoxIndex(gradientKernelComboBox, conf.gradient_2d);

	// Sekcja Buffer 1D kernels
	setComboBoxIndex(erodeKernelComboBox_2, conf.erode_1d);
	setComboBoxIndex(dilateKernelComboBox_2, conf.dilate_1d);
	setComboBoxIndex(gradientKernelComboBox_2, conf.gradient_1d);
	setComboBoxIndex(subtractKernelComboBox, conf.subtract_1d);
	setComboBoxIndex(hitmissMemTypeComboBox, conf.hitmiss_1d);
	datatypeComboBox->setCurrentIndex(conf.dataType);
}

Configuration Settings::configurationModel() const
{
	Configuration conf;

	// Sekcja [gui]
	conf.maxImageWidth = maxImageWidthLineEdit->text().toInt();
	conf.maxImageHeight = maxImageHeightLineEdit->text().toInt();
	conf.defaultImage = defaultImageLineEdit->text();

	// Sekcja [opencl]
	conf.atomicCounters = useAtomicCountersCheckBox->isChecked();
	conf.workgroupSizeX = workgroupSizeXComboBox->currentText().toInt();
	conf.workgroupSizeY = workgroupSizeYComboBox->currentText().toInt();

	/// Sekcja [kernel-buffer2D]
	conf.erode_2d = erodeKernelComboBox->currentText();
	conf.dilate_2d = dilateKernelComboBox->currentText();
	conf.gradient_2d = gradientKernelComboBox->currentText();

	// Sekcja [kernel-buffer1D]
	conf.erode_1d = erodeKernelComboBox_2->currentText();
	conf.dilate_1d = dilateKernelComboBox_2->currentText();
	conf.gradient_1d = gradientKernelComboBox_2->currentText();
	conf.subtract_1d = subtractKernelComboBox->currentText();
	conf.hitmiss_1d = hitmissMemTypeComboBox->currentText();
	conf.dataType = datatypeComboBox->currentIndex();

	return conf;
}
