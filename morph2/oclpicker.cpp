#include "oclpicker.h"

#include <QMessageBox>
#include <QDebug>

oclPicker::oclPicker(const PlatformDevicesMap& map,
	QWidget* parent)
	: QDialog(parent)
	, platformId(0)
	, deviceId(0)
{
	setupUi(this);

	QList<QTreeWidgetItem*> items;

	QMapIterator<oclPlatformDesc, QList<oclDeviceDesc> > i(map);
	while(i.hasNext())
	{
		i.next();

		QString cbText = QString("[%1] %2 %3")
				.arg(i.key().id)
				.arg(QString::fromStdString(i.key().name))
				.arg(QString::fromStdString(i.key().version));

		QTreeWidgetItem* pl = new QTreeWidgetItem((QTreeWidget*)0, QStringList(cbText));
		items.append(pl);

		auto& devlist = i.value();
		foreach(oclDeviceDesc desc, devlist)
		{
			QTreeWidgetItem* dev = new QTreeWidgetItem(pl,
				QStringList(QString::fromStdString(desc.name)));
			items.append(dev);

			QString description = QString(
				"Images supported: %1\n"
				"Compute units: %2\n"
				"Clock frequency: %3 MHz\n"
				"Constant buffer size: %4 B (%5 kB)\n"
				"Memory object allocation: %6 B (%7 MB)\n"
				"Local memory size: %8 B (%9 kB)\n"
				"Local memory type: %10\n")
					.arg(desc.imagesSupported ? "yes" : "no")
					.arg(desc.maxComputeUnits)
					.arg(desc.maxClockFreq)
					.arg(desc.maxConstantBufferSize)
					.arg(desc.maxConstantBufferSize >> 10)
					.arg(desc.maxMemAllocSize)
					.arg(desc.maxMemAllocSize >> 20)
					.arg(desc.localMemSize)
					.arg(desc.localMemSize >> 10)
					.arg((desc.localMemType == CL_LOCAL) ? "local" : "global");

			devToDesc.insert(QString::fromStdString(desc.name), description);
		}
	}

	textEdit->setPlainText("Choose a platform");

	treeWidget->insertTopLevelItems(0, items);
	treeWidget->header()->setStretchLastSection(false);
	treeWidget->header()->setResizeMode(QHeaderView::ResizeToContents);

	connect(treeWidget, SIGNAL(itemSelectionChanged()),
		SLOT(onItemSelectionChanged()));

	connect(pushButton, SIGNAL(pressed()),
		SLOT(onChoosePressed()));
}

oclPicker::~oclPicker()
{
}

void oclPicker::onItemSelectionChanged()
{
	QList<QTreeWidgetItem*> selected = treeWidget->selectedItems();
	pushButton->setEnabled(false);

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
			pushButton->setEnabled(true);
		}
	}
}

void oclPicker::onChoosePressed()
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

	QDialog::accept();
}
