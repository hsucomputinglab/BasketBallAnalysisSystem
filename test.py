from onvif import ONVIFCamera

camera = ONVIFCamera("10.30.3.28", 80, "nckusport", "Ncku1234")
device_info = camera.devicemgmt.GetDeviceInformation()
print(device_info)
