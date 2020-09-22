from flask import Flask,render_template,request
from keras.models import load_model
from keras.optimizers import Adam
from keras import models
import numpy as np
from keras.preprocessing import image
import os
app = Flask(__name__)
classes={'Tomato___Bacterial_spot': 0, 'Tomato___Early_blight': 1, 'Tomato___Late_blight': 2, 'Tomato___Leaf_Mold': 3, 'Tomato___Septoria_leaf_spot': 4, 'Tomato___Spider_mites Two-spotted_spider_mite': 5, 'Tomato___Target_Spot': 6, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 7, 'Tomato___Tomato_mosaic_virus': 8, 'Tomato___healthy': 9}
upload_folder="C:/Users/Vinit/Desktop/tomato/static"
def predict(img_loc):
	model=load_model('my_model.h5')
	model.compile(loss='categorical_crossentropy',
	              optimizer=Adam(lr=0.001),
	                  metrics=['accuracy'])
	img=image.load_img(img_loc,target_size=(128, 128))
	x=image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	images=np.vstack([x])
	for key,value in classes.items():
	    if(value==model.predict_classes(images,batch_size=1)):
	        return key


@app.route('/',methods=["GET","POST"])
def upload():
	if request.method=="POST":
		image_file=request.files["image"]
		if image_file:
			image_loc=os.path.join(
				upload_folder,image_file.filename
			)
			image_file.save(image_loc)
			disease=predict(image_loc)
			return render_template('tomatoindex2.html',prediction=disease,path=image_file.filename)
	return render_template('tomatoindex.html')


if __name__=='__main__':
	app.run(debug=True)