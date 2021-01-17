## Using the NDI_FaceTrack.exe

Launching the NDI_FaceTrack.exe (with no additional parameters) launches the full application.

The following arguments allow customization of the launch parameters.

**Hotkeys** 

- Tracking Enable/Disable (Global) - Ctrl+Shift+ID e.g Ctrl+Shift+2

- Reset Tracking (Local) - Alt+R

- Move Camera (Local) - WASD

- Zoom Camera (Local) - QE

## Sample Launch:

`NDI_FaceTrack.exe --name "BDCAM1 (PTZ100)" --enable_console False --id 2`

## Arguments:

`-n` `--name` Name: Use this option to connect directly to an NDI source upon launch. Provide the Name of the camera to connect to format: `NameXXX (DeviceXXX)` Default: `None`

`-c` `--enable_console` Interface: Use this option to hide the default interface. Default: `False`

 Warning: You must pass the `name` parameter when launching. |
 Warning: By using this option, you are unable to use any of the following features: |

- **Moveable Home Position**: Position where you want the tracker to place 'center' the target person.

- **Speed Sensitivity**: Control how fast the camera reacts to movement.

- **Horizontal Threshold**: The threshold on how far the target body is from the center. If the distance is above the threshold, horizontal camera tracking will start.

- **Vertical Threshold**: The threshold on how far the target body is from the center. If the distance is above the threshold, vertical camera tracking will start.

- **Change sources**: Use the menu bar to select available NDI Sources

Vertical Tracking toggle and AutoFind toggle are also not useable during hidden UI mode.

`-l` `--logging` Logging: Use this option for debugging purposes. By enabling this option, you will generate a `debug.log` on the root directory which will contain debug information. This file may get excessively large so please delete after a debugging session. Default: `False`

`-i` `--id` ID: Launch the application with an associated ID. Use this ID to control the global hotkeys. 
Example: --id = 3; Ctrl+Shift+3 = Enable/Disable Tracking


