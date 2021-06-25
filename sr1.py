import cv2

img=cv2.imread("test2.jpeg")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "ESPCN_x4.pb"
sr.readModel(path)
sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio
result = sr.upsample(img) # upscale the input image
cv2.imwrite("output1.png", result)
