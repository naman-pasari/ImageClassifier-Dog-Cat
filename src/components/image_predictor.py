from src.utils import *
import cv2

try:
    def predict(image_path):
        image= cv2.imread(image_path)
        resize = tf.image.resize(image, (256,256))
        new_model = load_model(os.path.join(os.getcwd(),'notebook',"models","imageclassifier.h5"))
        prediction=new_model.predict(np.expand_dims(resize/255, 0))
        print(prediction)
        if prediction > 0.5: 
            print(f'Predicted class is Dog')
        else:
            print(f'Predicted class is Cat')
        logging.info("Prediction successfull...")


except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    #Giving a custom image path
    image_path=os.path.join(os.getcwd(),'notebook','images','test_dog.jpg')
    predict(image_path=image_path)