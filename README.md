# metamaterials
Codes and instructions to control the setup that scans the acoustic modes of metamaterial models

## Codes
* [printer.py](https://github.com/jt-y/metamaterials-setup/blob/main/printer.py): Contains methods to control the 3D printer. Creates a class of objects named `Printer`.
* [srsamp.py](https://github.com/jt-y/metamaterials-setup/blob/main/srsamp.py): Contains methods to take data from the lock-in amplifier. Creates a class of objects called `LockinAmp`.
* [scanner.py](https://github.com/jt-y/metamaterials-setup/blob/main/scanner.py): Contains methods to scan the lattice and take data. Creates a class of objects called `Scanner`. `Scanner` has two attributes:
  1. `self.amp`: a `LockinAmp` object
  2. `self.p`: a `Printer` object


## Instructions
Installing the Linux GPIB driver: [instructions](https://github.com/jt-y/metamaterials-setup/blob/main/install%20GPIB%20driver.md)
