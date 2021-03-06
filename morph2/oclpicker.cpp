#include "oclpicker.h"

#include <QMessageBox>
#include <QDebug>

static QString q_deviceTypeToString(QCLDevice::DeviceTypes type)
{
	if(type & QCLDevice::GPU)
		return "GPU";
	if(type & QCLDevice::CPU)
		return "CPU";
	if(type & QCLDevice::Accelerator)
		return "Accelerator";

	return "Undefined";
}

static QString q_cacheTypeToString(QCLDevice::CacheType type)
{
	switch(type)
	{
	case QCLDevice::NoCache: return "No cache";
	case QCLDevice::ReadOnlyCache: return "Read-only cache";
	case QCLDevice::ReadWriteCache: return "Read-Write cache";
	default: return "Undefined";
	}
}

oclPicker::oclPicker(const PlatformDevicesList& list,
	QWidget* parent)
	: QDialog(parent)
	, platformId(0)
	, deviceId(0)
	, interop(false)
	, backend(OB_Images)
{
	setupUi(this);
	Q_ASSERT(buttonBox->button(QDialogButtonBox::Ok));
	Q_ASSERT(buttonBox->button(QDialogButtonBox::Cancel));

	buttonBox->button(QDialogButtonBox::Ok)->setText("Choose");
	buttonBox->button(QDialogButtonBox::Cancel)->setText("No OpenCL");

	connect(buttonBox, SIGNAL(accepted()), 
		SLOT(accept()));
	connect(buttonBox, SIGNAL(rejected()),
		SLOT(reject()));

	splitter->setStretchFactor(0, 7);
	splitter->setStretchFactor(1, 3);

	QList<QTreeWidgetItem*> items;

	QListIterator<QPair<QCLPlatform, QList<QCLDevice>>> i(list);
	while(i.hasNext())
	{
		auto pair = i.next();
		QCLPlatform p = pair.first;

		QString cbText = QString("%1 %2")
				.arg(p.name())
				.arg(p.version());

		QTreeWidgetItem* pl = new QTreeWidgetItem((QTreeWidget*)0, QStringList(cbText));
		items.append(pl);

		auto& devlist = pair.second;
		foreach(QCLDevice dev, devlist)
		{
			QCLDevice::DeviceTypes deviceType = dev.deviceType();
			bool hasImage2D = dev.hasImage2D();
			bool isGpu = deviceType & QCLDevice::GPU;

			QTreeWidgetItem* devItem = new QTreeWidgetItem(pl,
				QStringList(dev.name()));
			devItem->setData(0, Qt::UserRole, isGpu);
			devItem->setData(0, Qt::UserRole + 1, hasImage2D);
			items.append(devItem);
			
			int computeUnits = dev.computeUnits();
			int clockFrequency = dev.clockFrequency();
			
			quint64 maximumAllocationSize = dev.maximumAllocationSize();
			quint64 globalMemorySize = dev.globalMemorySize();
			QCLDevice::CacheType globalMemoryCacheType = dev.globalMemoryCacheType();
			quint64 globalMemoryCacheSize = dev.globalMemoryCacheSize();
			int globalMemoryCacheLineSize = dev.globalMemoryCacheLineSize();
			quint64 localMemorySize = dev.localMemorySize();
			bool isLocalMemorySeparate = dev.isLocalMemorySeparate();
			quint64 maximumConstantBufferSize = dev.maximumConstantBufferSize();

			QString description = QString(
				"Device type: %1\n"
				"Compute units: %2\n"
				"Clock frequency: %3 MHz\n"
				"Images 2D supported: %4\n"
				"Maximum allocation size: %5 B (%6 MB)\n"
				"Global memory size: %7 B (%8 MB)\n"
				"Global memory cache type: %9\n"
				"Global memory cache size: %10 B\n"
				"Global memory cache line size: %11\n"
				"Local memory size: %12 B (%13 kB)\n"
				"Local memory type: %14\n"
				"Constant buffer size: %15 B (%16 kB)\n")
					.arg(q_deviceTypeToString(deviceType))
					.arg(computeUnits)
					.arg(clockFrequency)
					.arg(hasImage2D ? "yes" : "no")
					.arg(maximumAllocationSize)
					.arg(maximumAllocationSize >> 20)
					.arg(globalMemorySize)
					.arg(globalMemorySize >> 20)
					.arg(q_cacheTypeToString(globalMemoryCacheType))
					.arg(globalMemoryCacheSize)
					.arg(globalMemoryCacheLineSize)
					.arg(localMemorySize)
					.arg(localMemorySize >> 10)
					.arg(isLocalMemorySeparate ? "on-chip" : "global")
					.arg(maximumConstantBufferSize)
					.arg(maximumConstantBufferSize >> 10);
			devToDesc.insert(dev.name(), description);
		}
	}

	textEdit->setPlainText("Choose a platform");

	treeWidget->insertTopLevelItems(0, items);
	treeWidget->header()->setStretchLastSection(false);
	treeWidget->header()->setResizeMode(QHeaderView::ResizeToContents);

	int platformsCount = treeWidget->topLevelItemCount();
	for(int i = 0; i < platformsCount; ++i)
	{
		QTreeWidgetItem* pl = treeWidget->topLevelItem(i);
		int devicesCount = pl->childCount();
		for(int j = 0; j < devicesCount; ++j)
		{
			QTreeWidgetItem* dev = pl->child(j);
			bool hasImages = dev->data(0, Qt::UserRole + 1).toBool();
			dev->setHidden(!hasImages);
		}
	}

	treeWidget->expandAll();

	connect(treeWidget, SIGNAL(itemSelectionChanged()),
		SLOT(onItemSelectionChanged()));
	connect(tryInteropCheckBox, SIGNAL(toggled(bool)),
		SLOT(onFilteringChanged()));
	connect(backendComboBox, SIGNAL(currentIndexChanged(int)),
		SLOT(onFilteringChanged()));

	if(treeWidget->topLevelItemCount() > 0)
	{
		QTreeWidgetItem* top = treeWidget->topLevelItem(0);
		int cc = top->childCount();
		for(int i = 0; i < cc; ++i)
		{
			auto child = top->child(i);
			if(!child->isHidden())
			{
				treeWidget->setCurrentItem(child);
				break;
			}			
		}
		
	}

	treeWidget->header()->setStretchLastSection(true);
}

oclPicker::~oclPicker()
{
}

void oclPicker::accept()
{
	QList<QTreeWidgetItem*> selected = treeWidget->selectedItems();

	// Nigdy nie powinno wystapic
	if(selected.isEmpty())
	{
		QMessageBox::critical(nullptr, "Morph OpenCL", "Please select a device");
		return;
	}
	else if(!selected[0]->parent())
	{
		QMessageBox::critical(nullptr, "Morph OpenCL", "Please select a device, not a platform");
		return;
	}

	QTreeWidgetItem* item = selected[0];
	QTreeWidgetItem* parent = item->parent();

	platformId = treeWidget->indexOfTopLevelItem(parent);
	deviceId = parent->indexOfChild(item);
	interop = tryInteropCheckBox->isChecked();
	backend = static_cast<EOpenCLBackend>(backendComboBox->currentIndex());

	QDialog::accept();
}

void oclPicker::onItemSelectionChanged()
{
	QList<QTreeWidgetItem*> selected = treeWidget->selectedItems();
	buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);

	if(selected.isEmpty())
	{
		textEdit->setPlainText("Choose a platform");
		return;
	}

	QTreeWidgetItem* item = selected[0];

	if(!item->parent())
	{
		QString s = QString("%1\n\nAvailable devices:\n")
			.arg(item->text(0));
		for(int i = 0; i < item->childCount(); ++i)
		{
			s += QString("* %1\n").arg(item->child(i)->text(0));
		}
		textEdit->setPlainText(s);
	}
	else
	{
		QString dev_name = item->text(0);
		QMap<QString, QString>::iterator i = devToDesc.find(dev_name);
		if(i != devToDesc.end())
		{
			textEdit->setPlainText(i.value());
			buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
		}
	}
}

void oclPicker::onFilteringChanged()
{
	QList<QTreeWidgetItem*> selectedItems = treeWidget->selectedItems();
	QTreeWidgetItem* selected = nullptr;
	if(!selectedItems.isEmpty())
		selected = selectedItems[0];

	bool forceRefresh = false;
	int platformsCount = treeWidget->topLevelItemCount();

	auto currentBackend = static_cast<EOpenCLBackend>
		(backendComboBox->currentIndex());
	bool currentInterop = tryInteropCheckBox->isChecked();

	for(int i = 0; i < platformsCount; ++i)
	{
		QTreeWidgetItem* pl = treeWidget->topLevelItem(i);
		int devicesCount = pl->childCount();
		for(int j = 0; j < devicesCount; ++j)
		{
			QTreeWidgetItem* dev = pl->child(j);

			bool isGpu = dev->data(0, Qt::UserRole).toBool();
			bool hasImages = dev->data(0, Qt::UserRole + 1).toBool();

			bool toHide = false;
			if(currentBackend == OB_Images)
				toHide |= !hasImages;
			if(currentInterop)
				toHide |= !isGpu;

			if(toHide && (dev == selected))
				forceRefresh = true;
			dev->setHidden(toHide);	
		}
	}

	if(forceRefresh)
	{
		// Zalozenie jest takie ze jakas platforma istnieje
		// Oto dba Controller
		treeWidget->setItemSelected(selected, false);
		treeWidget->setItemSelected(treeWidget->topLevelItem(0), true);		
	}
}
