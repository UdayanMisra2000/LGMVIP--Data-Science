import cv2

img = cv2.imread (r'C:\Users\udaya\Downloads\super_car.jpg')
gr_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur_gr_img = cv2.GaussianBlur(gr_img, (21,21),0)

pencil_sketch_img = cv2.divide(gr_img, blur_gr_img , scale=256)

cv2.imshow('pencil sketch',pencil_sketch_img)
cv2.imshow('Car',img)
cv2.waitKey(0)
