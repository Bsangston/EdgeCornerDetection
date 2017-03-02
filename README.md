# EdgeCornerDetection
Canny Edge Detector &amp; Harris Corner Detector implementation in Python

#Canny.py
Takes in one required command line parameter for image file path, and a second optional paramter to set threshold. This threshold value is a ratio of the max pixel value in the thinned edge image, so it should be between [0, 1.0]. Default value is 0.33, and the lower threshold is set at a ratio of 1:2. Images are handled as 32bit floats. Program accelerated by compiling with numba.jit.

Three example outputs are included along with their original images:  
  -flower.jpg    
  -canny_flower.png   
  -car.jpg  
  -canny_car.png  
  -canterbury.jpg   
  -canny_canterbury.png 

#Harris.py 
Takes in one required command line parameter for image file path, and a second optional paramter to set threshold. Thresholds may vary wildly from image to image. Default is 0.55 (for use with the flower image). Blue crosshairs are drawn to represent detected corners. 

Note that non-maximum thresholding was not enough to prevent hundreds of detected corners in most images. Using the minimum eigenvalue method, raising the threshold to counteract this results in unwanted clustering of corner detected areas. Thus, a scalar modifies the corners that are draw on the screen such that only 25% of detected corners are accepted. This is an unelegant solution, but also drastically reduces the run time of the program, which is admittedly too long.

Three example outputs are included along with their original images:  
-flower.jpg			
-mona.jpg		  
-bubbles.jpg	  
-harris_flower.png    
-harris_mona.png    
-harris_bubbles.png 
