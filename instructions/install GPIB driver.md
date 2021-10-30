# Installing the Linux-GPIB driver

The GPIB-USB converter that I used is Agilent Technologies 82357B. I believe these instructions also work for Agilent Technologies 82357A.

## Install Ubuntu
If your computer is not running the Linux operating system, download and install Ubuntu following the instructions [here](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview). 

## Download the installation package
Download the **latest** version of the Linux GPIB driver from [here](https://sourceforge.net/projects/linux-gpib/files/linux-gpib%20for%203.x.x%20and%202.6.x%20kernels/).

## Unpack the installation package
The downloaded file is a .tar.gz file, which is similar to a .zip file. Move the package to the desktop and untar it with the following command in the terminal:

```
tar -xzf <filename.tar.gz>
```
Navigate into this directory that you just untarred. There should be two .tar.gz files and a README file. Use the same command to untar the two .tar.gz files. 

## Install user space
The two directories that you just untarred are installation packages for the user and kernel space respectively. You can tell them apart from their names. Navigate into the gpib-user directory. Run the following commands. Enter your password when prompted.

```
sudo ./configure
sudo make
sudo make install
```
## Install kernel space
Navigate into the gpib-kernel directory. Run the following commands. 

```
sudo make
sudo make install
```

## Download firmware
Navigate back to your desktop. Run the following command to download the firmware for 82357B/82357A to your desktop. Untar it with the command mentioned above. 
```
wget --content-disposition --no-check-certificate http://linux-gpib.sourceforge.net/firmware/gpib_firmware-2008-08-10.tar.gz
```
Download the fxload file to your desktop. It is used to upload the firmware to the GPIB interface.
```
wget --content-disposition --no-check-certificate https://downloads.sourceforge.net/project/linux-hotplug/fxload/2008_10_13/fxload-2008_10_13.tar.gz
```
Untar it and install it.
```
tar xvfz fxload-2008_10_13.tar.gz
cd fxload-2008_10_13
sudo make
sudo make install
```

## Edit gpib.conf
Navigate to /usr/local/etc. Use the `ls` command to see the files and directories in here. There should be a file named "gpib.conf" and a directory named "udev" Run the following command.

```
sudo nano gpib.conf
```
This opens the gpib.conf file. Find the line that specifies the board type of the first interface and change the board type to "agilent_82357a"

```
interface {
        board_type = "agilent_82357a"
        name = "agi"
        ...
}
```

Then load the kernel modules by running:

```
sudo modprobe gpib_common
sudo modprobe agilent_82357a
```

## Establish connection between the device and your computer

Plug the GPIB-USB converter into your computer. Run `lsusb`. You should see a list of devices. Find the line that corresponds to your GPIB-USB converter. That should look something like this:

```
...
Bus 001 Device 005: ID 0957:0718 Agilent Technologies, Inc. 82357B ()
...
```
Take note of the bus and device numbers. In the above example, the Bus# is 001 and the Device# is 005. Run the following command with Bus# and Device# set appropriately.

```
sudo fxload -D /dev/bus/usb/Bus#/Device#  -t fx2 -I /home/metamaterials/Desktop/gpib_firmware-2008-08-10/agilent_82357a/measat_releaseX1.8.hex 
```
Run `lsusb` again and confirm the Device# has increased by 1.

Run the following command again with the new Bus# and Device# set appropriately.

```
sudo fxload -D /dev/bus/usb/Bus#/Device#  -t fx2 -I /home/metamaterials/Desktop/gpib_firmware-2008-08-10/agilent_82357a/measat_releaseX1.8.hex 
```

The 3 LEDs on the adaptor should be lit.

Change permissions on `/dev/gpib0`:

```
sudo chmod 777 /dev/gpib0
```

Now, initialize the dongle. `gpib_config` has some trouble finding the library, so create a symbolic link first:

```
sudo ln -s /usr/local/lib/libgpib.so.0 /lib/libgpib.so.0
sudo gpib_config
```

Now only the green LED should be lit. This means that you have successfully established connection between your computer and the device!

## Uninstall the driver

Detele
```
/usr/local/etc/gpib.conf
/usr/local/etc/udev/rules.d
```

## Useful resources
1. [Basic Linux command lines](https://ubuntu.com/tutorials/command-line-for-beginners#3-opening-a-terminal)
2. [GPIB interfacing using Agilent 82357B on Ubuntu Linux](https://gist.github.com/turingbirds/6eb05c9267a6437183a9567700e8581a)
3. [Minhal's instructions on setting up the GPIB driver](https://docs.google.com/document/d/17xDrLCJeWzFlLrQEiko42a_ro7sBEtSmDbPg0o_xBlQ/edit?usp=sharing)
