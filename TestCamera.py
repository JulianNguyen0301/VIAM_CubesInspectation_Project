import neoapi
import cv2
camera = neoapi.Cam()
camera.Connect()   
camera.f.ExposureTime.Set(1000)
camera.f.Width.Set(2592)
camera.f.Height.Set(2048)

camera.f.OffsetX.Set(0)
camera.f.Gain.Set(0)
camera.f.PixelFormat.Set(neoapi.PixelFormat_BayerRG8)
camera.SetSynchronFeatureMode(False) 

#Tắt sharpending
camera.f.TriggerMode.value = neoapi.TriggerMode_On
trigger = camera.f.TriggerSoftware
if camera.HasFeature("SharpeningEnable"):
    camera.SetFeature("SharpeningEnable", False)
if camera.HasFeature("NoiseReductionEnable"):
    camera.SetFeature("NoiseReductionEnable", False)

#Bật chế độ use optimal buffer size
if camera.HasFeature("UseOptimalBufferSize"):
    camera.SetFeature("UseOptimalBufferSize", True)
img_camera = camera.GetImage()
cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
cv2.imshow("finalImg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()